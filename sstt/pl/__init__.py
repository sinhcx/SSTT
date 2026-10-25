"""Plotting functions for SSTT workflows."""

from .basic_plot import spatial_sampling_preview, celltype_composition_stack
from .spatial_plot import (
    spatial_network,
    spatial_polygons,
    spatial_bins,
    spatial_transcripts,
)
from .niche_plot import niche_enrichment_heatmap
from .gradient_plot import (
    selected_border_curves,
    border_normalized_distance,
    celltype_composition_trend,
    celltype_composition_consensus,
    distance_trend,
    fuzzy_clusters,
)
from .neighbor_enrichment_plot import (
    plot_component_distribution,
    plot_component_closure_index,
    plot_multihop_distribution_from_membership,
    plot_multihop_enrichment_heatmap,
    plot_observed_multihop_enrichment_heatmap,
    plot_component_distance_ridges,
    compute_cellwise_spatial_density,
    plot_density_pair_panel,
    plot_multihop_gene_trajectory,
    plot_multihop_genes_trajectory,
)
from .nmf_plot import (
    plot_cnmf_heatmap,
    plot_mp_gene_heatmap,
    plot_mp_go_dotplot_refined,
    plot_single_mp_go_barplot,
    plot_glm_heatmap,
)
from .cell_cell_interaction_plot import (
    plot_cellchat_style_bubble,
    draw_interaction_comparison_filtered,
    visualize_communication_high_res,
)

__all__ = [
    "spatial_sampling_preview",
    "celltype_composition_stack",
    "spatial_network",
    "spatial_polygons",
    "spatial_bins",
    "spatial_transcripts",
    "niche_enrichment_heatmap",
    "selected_border_curves",
    "border_normalized_distance",
    "celltype_composition_trend",
    "celltype_composition_consensus",
    "distance_trend",
    "fuzzy_clusters",
    "plot_component_distribution",
    "plot_component_closure_index",
    "plot_multihop_distribution_from_membership",
    "plot_multihop_enrichment_heatmap",
    "plot_observed_multihop_enrichment_heatmap",
    "plot_component_distance_ridges",
    "compute_cellwise_spatial_density",
    "plot_density_pair_panel",
    "plot_cnmf_heatmap",
    "plot_mp_gene_heatmap",
    "plot_mp_go_dotplot_refined",
    "plot_single_mp_go_barplot",
    "plot_glm_heatmap",
    "plot_cellchat_style_bubble",
    "draw_interaction_comparison_filtered",
    "visualize_communication_high_res",
    "plot_multihop_gene_trajectory",
    "plot_multihop_genes_trajectory",
]
