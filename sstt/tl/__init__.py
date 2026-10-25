"""Analysis tools for SSTT workflows."""

from .Interactive_select_ROI import InteractiveCurveSelector
from .gradient_binning import calculate_normalized_distances
from .fuzzy_clustering import (
    create_gene_distance_matrix,
    filter_spatially_variable_genes,
    run_fuzzy_clustering,
    run_fuzzy_cluster_enrichment,
)
from .neighbor_enrichment import (
    build_candidate_multihop_neighborhoods,
    filter_components_by_closure,
    filter_components_manually,
    build_component_distance_data,
    run_multihop_enrichment,
    compute_multihop_gene_gradient,
)
from .nmf import (
    run_cnmf_meta_analysis,
    run_mp_go_enrichment,
    compute_stable_meta_program_scores,
    build_distance_weighted_exposure_matrix,
    calculate_glm_associations,
)

__all__ = [
    "InteractiveCurveSelector",
    "calculate_normalized_distances",
    "create_gene_distance_matrix",
    "filter_spatially_variable_genes",
    "run_fuzzy_clustering",
    "run_fuzzy_cluster_enrichment",
    "build_candidate_multihop_neighborhoods",
    "filter_components_by_closure",
    "filter_components_manually",
    "build_component_distance_data",
    "run_multihop_enrichment",
    "run_cnmf_meta_analysis",
    "run_mp_go_enrichment",
    "compute_stable_meta_program_scores",
    "build_distance_weighted_exposure_matrix",
    "calculate_glm_associations",
    "compute_multihop_gene_gradient",
]
