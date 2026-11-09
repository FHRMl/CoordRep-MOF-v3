"""CoordRep-MOF: Periodic coordination priors and exact Procrustes polyhedral descriptors (CN=2..9)."""
from .geometry import DONOR_Z, METALS, validate_geometry
from .v1 import (
    TEMPLATE_BY_CN,
    TEMPLATE_DIRECTIONS,
    TEMPLATES,
    encode_coordination,
    extract_coordination,
    shape_scores,
)
from .v2 import (
    DEFAULT_CARBOXYLATE_C_CUTOFF,
    METHOD_ID_V2,
    encode_center_from_diagnostics,
    extract_coordination_v2,
    filter_carboxylate_c_donors,
)
from .v3 import (
    METHOD_ID_V3,
    TEMPLATE_BY_CN_V3,
    TEMPLATE_DIRECTIONS_V3,
    TEMPLATES_V3,
    CoordinationScalerV3,
    batched_high_cn_shape_scores,
    encode_coordination_v3,
    extract_coordination_v3,
    shape_scores_v3,
    valid_coordination_channels_v3,
)

__version__ = "0.3.0"
__all__ = [
    "DONOR_Z",
    "METALS",
    "validate_geometry",
    "TEMPLATE_BY_CN",
    "TEMPLATE_DIRECTIONS",
    "TEMPLATES",
    "shape_scores",
    "encode_coordination",
    "extract_coordination",
    "DEFAULT_CARBOXYLATE_C_CUTOFF",
    "METHOD_ID_V2",
    "filter_carboxylate_c_donors",
    "extract_coordination_v2",
    "encode_center_from_diagnostics",
    "METHOD_ID_V3",
    "TEMPLATE_BY_CN_V3",
    "TEMPLATE_DIRECTIONS_V3",
    "TEMPLATES_V3",
    "shape_scores_v3",
    "batched_high_cn_shape_scores",
    "encode_coordination_v3",
    "extract_coordination_v3",
    "valid_coordination_channels_v3",
    "CoordinationScalerV3",
]
