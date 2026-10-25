from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter

__all__ = [
    "plot_cellchat_style_bubble",
    "draw_interaction_comparison_filtered",
    "visualize_communication_high_res",
]


def plot_cellchat_style_bubble(
    df_results,
    score_col="Base_Score",
    sender_filter=None,
    receiver_filter=None,
    top_n=30,
    p_threshold=0.05,
    n_permutations=None,
    bubble_scale=30,
    max_bubble_factor=4.0,
    size_power=0.8,
    cmap="Spectral_r",
    figsize=None,
    xtick_fontsize=10,
    ytick_fontsize=10,
    xlabel_fontsize=11,
    ylabel_fontsize=11,
    title_fontsize=12,
    colorbar_fontsize=10,
    legend_fontsize=10,
    legend_title_fontsize=10,
    x_rotation=45,
    save_path=None,
    dpi=300,
):
    if df_results is None or len(df_results) == 0:
        print("提示: 输入结果为空。")
        return None
    if n_permutations is None:
        raise ValueError(
            "必须指定 n_permutations，并与计算 P_value 时使用的随机次数一致。"
        )
    n_permutations = int(n_permutations)
    if n_permutations < 1:
        raise ValueError("n_permutations 必须 >= 1。")
    if not 0 < p_threshold < 1:
        raise ValueError("p_threshold 必须位于 0 和 1 之间。")
    if bubble_scale <= 0:
        raise ValueError("bubble_scale 必须 > 0。")
    if max_bubble_factor < 1:
        raise ValueError("max_bubble_factor 必须 >= 1。")
    if size_power <= 0:
        raise ValueError("size_power 必须 > 0。")
    required_columns = {"Interaction", "Sender_Receiver", "P_value", score_col}
    missing_columns = required_columns.difference(df_results.columns)
    if missing_columns:
        raise ValueError("df_results 缺少必要列: " + ", ".join(sorted(missing_columns)))
    p_min = 1.0 / (n_permutations + 1.0)
    if p_min > p_threshold:
        raise ValueError(
            f"n_permutations={n_permutations} 时最小可达 P 值为 {p_min:.6g}，高于 p_threshold={p_threshold}。"
        )
    df_plot = df_results.copy()
    if "N_Permutations" in df_plot.columns:
        recorded_n = pd.to_numeric(df_plot["N_Permutations"], errors="coerce")
        recorded_n = np.unique(
            recorded_n[np.isfinite(recorded_n) & (recorded_n > 0)].astype(int)
        )
        if len(recorded_n) == 1 and recorded_n[0] != n_permutations:
            raise ValueError(
                f"绘图参数 n_permutations={n_permutations}，但结果表记录的 N_Permutations={recorded_n[0]}。"
            )
        if len(recorded_n) > 1:
            raise ValueError(
                "结果表中存在多个 N_Permutations。请按照随机次数拆分结果后分别绘制。"
            )
    df_plot["P_value"] = pd.to_numeric(df_plot["P_value"], errors="coerce")
    df_plot[score_col] = pd.to_numeric(df_plot[score_col], errors="coerce")
    df_plot = df_plot.dropna(subset=["P_value", score_col])
    below_floor = df_plot["P_value"] < p_min - 1e-12
    if below_floor.any():
        observed_min = df_plot.loc[below_floor, "P_value"].min()
        raise ValueError(
            f"结果中存在 P_value={observed_min:.6g}，低于 n_permutations={n_permutations} 对应的理论最小值 {p_min:.6g}。请检查随机次数或 P-value 计算公式。"
        )
    df_plot = df_plot[df_plot["P_value"] <= p_threshold].copy()
    if df_plot.empty:
        print(f"提示: 没有符合 P_value <= {p_threshold} 的交互数据。")
        return None
    if {"Sender", "Receiver"}.issubset(df_plot.columns):
        df_plot["_plot_sender"] = df_plot["Sender"].astype(str).str.strip()
        df_plot["_plot_receiver"] = df_plot["Receiver"].astype(str).str.strip()
    else:
        delimiter = None
        possible_delimiters = [" -> ", "->", " → ", "→", " - "]
        sender_receiver_text = df_plot["Sender_Receiver"].astype(str)
        for current_delimiter in possible_delimiters:
            if sender_receiver_text.str.contains(current_delimiter, regex=False).any():
                delimiter = current_delimiter
                break
        if delimiter is None:
            raise ValueError(
                "无法从 Sender_Receiver 自动识别 Sender 和 Receiver。建议结果表中保留 Sender 和 Receiver 两列。"
            )
        temp_split = sender_receiver_text.str.split(
            delimiter, n=1, expand=True, regex=False
        )
        if temp_split.shape[1] < 2:
            raise ValueError("Sender_Receiver 拆分失败。")
        df_plot["_plot_sender"] = temp_split[0].astype(str).str.strip()
        df_plot["_plot_receiver"] = temp_split[1].astype(str).str.strip()
    if sender_filter is not None:
        if isinstance(sender_filter, str):
            sender_filter = [sender_filter]
        df_plot = df_plot[df_plot["_plot_sender"].isin(sender_filter)].copy()
    if receiver_filter is not None:
        if isinstance(receiver_filter, str):
            receiver_filter = [receiver_filter]
        df_plot = df_plot[df_plot["_plot_receiver"].isin(receiver_filter)].copy()
    if df_plot.empty:
        print("提示: Sender/Receiver 过滤后没有符合条件的数据。")
        return None
    df_plot = df_plot.sort_values([score_col, "P_value"], ascending=[False, True])
    if top_n is not None:
        df_plot = df_plot.head(top_n).copy()
    candidate_p_levels = [p_threshold, 0.01, 0.001, 0.0001, 1e-05, 1e-06]
    p_levels = sorted(
        {
            float(level)
            for level in candidate_p_levels
            if level <= p_threshold + 1e-15 and level >= p_min - 1e-15
        },
        reverse=True,
    )
    if len(p_levels) == 0:
        p_levels = [p_threshold]
    level_index = _p_to_level_index(df_plot["P_value"].to_numpy(), p_levels)
    if len(p_levels) == 1:
        normalized_level = np.zeros_like(level_index, dtype=float)
    else:
        normalized_level = level_index / (len(p_levels) - 1)
    df_plot["_bubble_size"] = bubble_scale * (
        1.0 + (max_bubble_factor - 1.0) * normalized_level**size_power
    )
    x_order = pd.unique(df_plot["Sender_Receiver"].astype(str)).tolist()
    y_order = pd.unique(df_plot["Interaction"].astype(str)).tolist()
    df_plot["Sender_Receiver"] = pd.Categorical(
        df_plot["Sender_Receiver"].astype(str), categories=x_order, ordered=True
    )
    df_plot["Interaction"] = pd.Categorical(
        df_plot["Interaction"].astype(str),
        categories=list(reversed(y_order)),
        ordered=True,
    )
    n_x = len(x_order)
    n_y = len(y_order)
    if figsize is None:
        plot_figsize = (max(8, n_x * 0.8), max(6, n_y * 0.4))
    else:
        plot_figsize = tuple(figsize)
    fig, ax = plt.subplots(figsize=plot_figsize)
    scatter = ax.scatter(
        x=df_plot["Sender_Receiver"],
        y=df_plot["Interaction"],
        c=df_plot[score_col],
        s=df_plot["_bubble_size"],
        cmap=cmap,
        edgecolors="black",
        linewidths=0.6,
        alpha=0.9,
        zorder=3,
    )
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=15, pad=0.03)
    colorbar.set_label(
        score_col.replace("_", " "), fontsize=colorbar_fontsize, fontweight="bold"
    )
    colorbar.ax.tick_params(labelsize=colorbar_fontsize)
    legend_level_index = np.arange(len(p_levels), dtype=float)
    if len(p_levels) == 1:
        legend_normalized = np.zeros_like(legend_level_index)
    else:
        legend_normalized = legend_level_index / (len(p_levels) - 1)
    legend_sizes = bubble_scale * (
        1.0 + (max_bubble_factor - 1.0) * legend_normalized**size_power
    )
    legend_labels = []
    for index, cutoff in enumerate(p_levels):
        if index < len(p_levels) - 1:
            lower_cutoff = p_levels[index + 1]
            legend_labels.append(f"{lower_cutoff:g} < P ≤ {cutoff:g}")
        else:
            legend_labels.append(f"P ≤ {cutoff:g}")
    legend_handles = [
        ax.scatter(
            [], [], s=size, color="grey", edgecolors="black", linewidths=0.6, alpha=0.65
        )
        for size in legend_sizes
    ]
    ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        title=f"Empirical P-value\n{n_permutations} permutations",
        loc="upper left",
        bbox_to_anchor=(1.15, 1.0),
        frameon=False,
        labelspacing=1.2,
        fontsize=legend_fontsize,
        title_fontsize=legend_title_fontsize,
    )
    ax.tick_params(axis="x", labelrotation=x_rotation, labelsize=xtick_fontsize)
    for tick_label in ax.get_xticklabels():
        tick_label.set_horizontalalignment("right")
    ax.tick_params(axis="y", labelsize=ytick_fontsize)
    ax.grid(axis="both", linestyle=":", alpha=0.5, zorder=0)
    ax.set_title(
        f"Top {len(df_plot)} LR Interactions",
        fontsize=title_fontsize,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Cell Pairs (Sender → Receiver)", fontsize=xlabel_fontsize)
    ax.set_ylabel("Ligand–Receptor Pairs", fontsize=ylabel_fontsize)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    return ax


def draw_interaction_comparison_filtered(
    df_results,
    score_col="Contact_Score",
    focus_cells=None,
    p_threshold=0.05,
    top_n=None,
    figsize=(16, 8),
):
    base_size = figsize[1]
    node_size = base_size**2 * 12
    weight_scale = base_size * 1.2
    font_s = base_size * 1.5
    df_sig = df_results[df_results["P_value"] < p_threshold].copy()
    if "->" in df_sig["Sender_Receiver"].iloc[0]:
        df_sig[["Sender", "Receiver"]] = df_sig["Sender_Receiver"].str.split(
            "->", expand=True
        )
    else:
        df_sig[["Sender", "Receiver"]] = df_sig["Sender_Receiver"].str.split(
            "\\s*->\\s*", expand=True, regex=True
        )
    df_sig["Sender"] = df_sig["Sender"].str.strip()
    df_sig["Receiver"] = df_sig["Receiver"].str.strip()
    df_sig["pair_key"] = df_sig.apply(_interaction_pair_key, axis=1)
    df_count = df_sig.groupby("pair_key").size().reset_index(name="weight")
    df_strength = df_sig.groupby("pair_key")[score_col].sum().reset_index(name="weight")
    for df in [df_count, df_strength]:
        df[["Node1", "Node2"]] = df["pair_key"].str.split("-", expand=True)
    if focus_cells:
        if isinstance(focus_cells, str):
            focus_cells = [focus_cells]
        df_count = df_count[
            df_count["Node1"].isin(focus_cells) | df_count["Node2"].isin(focus_cells)
        ]
        df_strength = df_strength[
            df_strength["Node1"].isin(focus_cells)
            | df_strength["Node2"].isin(focus_cells)
        ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    _plot_aggregate_ax(
        ax1, df_count, "Number of Interactions", df_sig, weight_scale, node_size, font_s
    )
    _plot_aggregate_ax(
        ax2,
        df_strength,
        "Interaction Strength",
        df_sig,
        weight_scale,
        node_size,
        font_s,
    )
    plt.tight_layout()


def visualize_communication_high_res(
    adata,
    ligand_gene,
    receptor_genes,
    sender_types,
    receiver_types,
    analysis_adata=None,
    domain_name=None,
    domain_col="stage",
    celltype_col="celltype",
    spatial_key="spatial",
    effective_range=100.0,
    grid_res=400,
    intensity_threshold=0.01,
    dot_size=2.0,
    rcv_alpha=0.7,
    background_dot_size=0.5,
    background_alpha=0.1,
    figsize=(10, 9),
    ligand_cmap="YlGnBu",
    receptor_cmap="Reds",
    dpi=300,
    save_path=None,
    show=True,
):
    if isinstance(sender_types, str):
        sender_types = [sender_types]
    else:
        sender_types = list(sender_types)
    if isinstance(receiver_types, str):
        receiver_types = [receiver_types]
    else:
        receiver_types = list(receiver_types)
    if isinstance(receptor_genes, str):
        receptor_genes = [receptor_genes]
    else:
        receptor_genes = list(receptor_genes)
    if analysis_adata is None:
        analysis_adata = adata
    if effective_range <= 0:
        raise ValueError("`effective_range` must be > 0.")
    if grid_res < 10:
        raise ValueError("`grid_res` must be >= 10.")
    if spatial_key not in adata.obsm:
        raise KeyError(f"`adata.obsm[{spatial_key!r}]` was not found.")
    if spatial_key not in analysis_adata.obsm:
        raise KeyError(f"`analysis_adata.obsm[{spatial_key!r}]` was not found.")
    if celltype_col not in analysis_adata.obs:
        raise KeyError(f"`analysis_adata.obs[{celltype_col!r}]` was not found.")
    genes_needed = [ligand_gene, *receptor_genes]
    missing_genes = [
        gene for gene in genes_needed if gene not in analysis_adata.var_names
    ]
    if missing_genes:
        raise KeyError(f"Genes were not found in analysis_adata: {missing_genes}")
    full_coords = np.asarray(adata.obsm[spatial_key], dtype=float)[:, :2]
    if not np.isfinite(full_coords).all():
        raise ValueError("`adata` spatial coordinates contain non-finite values.")
    x_min, y_min = full_coords.min(axis=0)
    x_max, y_max = full_coords.max(axis=0)
    if domain_name is not None:
        if domain_col not in analysis_adata.obs:
            raise KeyError(f"`analysis_adata.obs[{domain_col!r}]` was not found.")
        domain_mask = analysis_adata.obs[domain_col].astype(str).to_numpy() == str(
            domain_name
        )
        comm_adata = analysis_adata[domain_mask].copy()
    else:
        comm_adata = analysis_adata.copy()
    if comm_adata.n_obs == 0:
        raise ValueError("No communication-analysis cells remain after filtering.")
    comm_coords = np.asarray(comm_adata.obsm[spatial_key], dtype=float)[:, :2]
    ligand_expr = comm_adata[:, ligand_gene].X
    if sp.issparse(ligand_expr):
        ligand_expr = ligand_expr.toarray()
    ligand_expr = np.asarray(ligand_expr).ravel()
    sender_mask = comm_adata.obs[celltype_col].astype(str).isin(
        sender_types
    ).to_numpy() & (ligand_expr > 0)
    if not sender_mask.any():
        raise ValueError(
            f"No {ligand_gene}-positive sender cells were found for sender_types={sender_types}."
        )
    sender_coords_np = comm_coords[sender_mask]
    sender_vals_np = ligand_expr[sender_mask]
    xi = np.linspace(x_min, x_max, grid_res)
    yi = np.linspace(y_min, y_max, grid_res)
    grid_x, grid_y = np.meshgrid(xi, yi)
    grid_points_np = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_points = torch.as_tensor(grid_points_np, dtype=torch.float32, device=device)
    sender_coords = torch.as_tensor(
        sender_coords_np, dtype=torch.float32, device=device
    )
    sender_vals = torch.as_tensor(sender_vals_np, dtype=torch.float32, device=device)
    # Render the untruncated Gaussian field over the full tissue extent.
    sigma = float(effective_range) / 3.0
    field_res = torch.zeros(len(grid_points), dtype=torch.float32, device=device)
    chunk_size = 5000
    with torch.no_grad():
        for start in range(0, len(grid_points), chunk_size):
            end = min(start + chunk_size, len(grid_points))
            dists_sq = torch.cdist(grid_points[start:end], sender_coords) ** 2
            weights = torch.exp(-dists_sq / (2.0 * sigma**2))
            field_res[start:end] = weights @ sender_vals
    grid_z = field_res.detach().cpu().numpy().reshape(grid_res, grid_res)
    del (grid_points, sender_coords, sender_vals, field_res)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    grid_z = gaussian_filter(grid_z, sigma=1.2)
    max_field = np.nanmax(grid_z)
    if np.isfinite(max_field) and max_field > 0:
        grid_z_masked = np.where(
            grid_z >= max_field * intensity_threshold, grid_z, np.nan
        )
    else:
        grid_z_masked = np.full_like(grid_z, np.nan)
    receiver_mask = (
        comm_adata.obs[celltype_col].astype(str).isin(receiver_types).to_numpy()
    )
    receptor_expr = comm_adata[:, receptor_genes].X
    if sp.issparse(receptor_expr):
        receptor_expr = receptor_expr.toarray()
    receptor_expr = np.asarray(receptor_expr, dtype=float)
    all_subunits_positive = np.all(receptor_expr > 0, axis=1)
    receptor_score = np.where(all_subunits_positive, receptor_expr.mean(axis=1), 0.0)
    final_receiver_mask = receiver_mask & (receptor_score > 0)
    receiver_coords = comm_coords[final_receiver_mask]
    receiver_values = receptor_score[final_receiver_mask]
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor="white")
    ax.scatter(
        full_coords[:, 0],
        full_coords[:, 1],
        c="lightgray",
        s=background_dot_size,
        alpha=background_alpha,
        linewidth=0,
        rasterized=True,
        zorder=1,
    )
    if not np.all(np.isnan(grid_z_masked)):
        cf = ax.contourf(
            grid_x,
            grid_y,
            grid_z_masked,
            levels=30,
            cmap=ligand_cmap,
            alpha=0.6,
            antialiased=True,
            zorder=2,
        )
        ax.contour(
            grid_x,
            grid_y,
            grid_z_masked,
            levels=30,
            colors="white",
            linewidths=0.05,
            alpha=0.2,
            zorder=2,
        )
        cbar_field = fig.colorbar(cf, ax=ax, shrink=0.4, pad=0.02, aspect=30)
        cbar_field.outline.set_visible(False)
        cbar_field.set_label(f"{ligand_gene} Field", fontsize=10)
    if len(receiver_coords) > 0:
        receptor_scatter = ax.scatter(
            receiver_coords[:, 0],
            receiver_coords[:, 1],
            c=receiver_values,
            s=dot_size,
            cmap=receptor_cmap,
            edgecolors="none",
            linewidth=0,
            alpha=rcv_alpha,
            zorder=3,
        )
        cbar_receptor = fig.colorbar(
            receptor_scatter, ax=ax, shrink=0.4, pad=0.08, aspect=30
        )
        cbar_receptor.outline.set_visible(False)
        cbar_receptor.set_label("Receptor Exp", fontsize=10)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    receptor_label = ", ".join(receptor_genes)
    if domain_name is None:
        title = f"{ligand_gene} → {receptor_label}"
    else:
        title = f"Domain: {domain_name}\n{ligand_gene} → {receptor_label}"
    ax.set_title(title, fontsize=12, pad=15)
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", transparent=True)
        print(f"Saved to: {save_path}")
    if show:
        plt.show()
    return ax


def _p_to_level_index(p_values, p_levels):
    p_values = np.asarray(p_values, dtype=float)
    level_index = np.zeros(len(p_values), dtype=int)
    for index, cutoff in enumerate(p_levels):
        level_index[p_values <= cutoff] = index
    return level_index


def _interaction_pair_key(row):
    return "-".join(sorted([row["Sender"], row["Receiver"]]))


def _plot_aggregate_ax(ax, data, title, df_sig, weight_scale, node_size, font_s):
    all_nodes = sorted(list(set(df_sig["Sender"]) | set(df_sig["Receiver"])))
    num_nodes = len(all_nodes)
    angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
    node_coords = {node: (np.cos(a), np.sin(a)) for node, a in zip(all_nodes, angles)}
    colors = plt.cm.get_cmap("tab20")(np.linspace(0, 1, num_nodes))
    node_color_map = {node: colors[i] for i, node in enumerate(all_nodes)}
    if not data.empty:
        max_w = data["weight"].max()
        for _, row in data.iterrows():
            n1, n2 = (row["Node1"], row["Node2"])
            edge_w = row["weight"] / max_w * weight_scale
            color = node_color_map[n1]
            if n1 == n2:
                x, y = node_coords[n1]
                norm = np.sqrt(x**2 + y**2)
                ux, uy = (x / norm, y / norm)
                vx, vy = (-uy, ux)
                p1 = (x + 0.4 * ux + 0.2 * vx, y + 0.4 * uy + 0.25 * vy)
                p2 = (x + 0.4 * ux - 0.2 * vx, y + 0.4 * uy - 0.25 * vy)
                path = mpath.Path([(x, y), p1, p2, (x, y)], [1, 4, 4, 4])
                ax.add_patch(
                    mpatches.PathPatch(
                        path,
                        facecolor="none",
                        edgecolor=color,
                        lw=edge_w,
                        alpha=0.5,
                        zorder=1,
                    )
                )
            else:
                connection = f"arc3,rad=0.05"
                ax.annotate(
                    "",
                    xy=node_coords[n1],
                    xytext=node_coords[n2],
                    arrowprops=dict(
                        arrowstyle="-",
                        color=color,
                        connectionstyle=connection,
                        alpha=0.3,
                        linewidth=edge_w,
                    ),
                )
    for node, (x, y) in node_coords.items():
        ax.scatter(
            x, y, s=node_size, color=node_color_map[node], edgecolors="none", zorder=10
        )
        ha = "left" if x > 0.1 else "right" if x < -0.1 else "center"
        va = "bottom" if y > 0.1 else "top" if y < -0.1 else "center"
        ax.text(
            x * 1.2, y * 1.2, node, fontsize=font_s, fontweight="bold", ha=ha, va=va
        )
    ax.set_title(title, fontsize=font_s * 1.3, pad=20)
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.7, 1.7)
    ax.set_aspect("equal")
    ax.axis("off")
