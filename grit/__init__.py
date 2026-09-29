"""GRIT implementation utilities."""

from grit.curvature import (
    CurvatureCorrectionResult,
    curvature_corrected_preservation_gradients,
    hessian_vector_product,
    project_vector_with_module_projectors,
    trainable_named_parameters,
)
from grit.predictor import (
    PredictorStepInfo,
    forward_with_predictor_step,
    linear_weight_parameter_names,
    temporary_predictor_step,
)
from grit.projection import (
    ProjectorBuildResult,
    ProjectorAttachResult,
    apply_attached_gradient_projection,
    apply_gradient_projection,
    attach_projectors_to_modules,
    attached_projectors,
    build_projectors_from_covariances,
    collect_activation_covariances,
    load_projectors,
    projector_diagnostics,
)
from grit.preservation_loss import (
    PreservationLossResult,
    aggregate_preservation_loss,
    preservation_kl_loss,
)
from grit.trust_region import (
    TrustRegionProjection,
    build_sparse_support_mask,
    geometric_interpolation_log_probs,
    kl_from_log_probs,
    project_to_kl_ball,
    sparse_default_log_probs,
    token_kl_from_logits,
)
from grit.update import (
    GritUpdateConfig,
    GritUpdateResult,
    assemble_grit_update,
)

__all__ = [
    "CurvatureCorrectionResult",
    "GritUpdateConfig",
    "GritUpdateResult",
    "PredictorStepInfo",
    "ProjectorBuildResult",
    "PreservationLossResult",
    "ProjectorAttachResult",
    "TrustRegionProjection",
    "aggregate_preservation_loss",
    "assemble_grit_update",
    "apply_attached_gradient_projection",
    "apply_gradient_projection",
    "attach_projectors_to_modules",
    "attached_projectors",
    "build_sparse_support_mask",
    "build_projectors_from_covariances",
    "collect_activation_covariances",
    "curvature_corrected_preservation_gradients",
    "forward_with_predictor_step",
    "geometric_interpolation_log_probs",
    "hessian_vector_product",
    "kl_from_log_probs",
    "linear_weight_parameter_names",
    "load_projectors",
    "preservation_kl_loss",
    "project_to_kl_ball",
    "project_vector_with_module_projectors",
    "projector_diagnostics",
    "sparse_default_log_probs",
    "temporary_predictor_step",
    "token_kl_from_logits",
    "trainable_named_parameters",
]
