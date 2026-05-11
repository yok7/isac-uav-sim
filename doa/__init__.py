"""DOA algorithms package.

Provides:
- Classical DOA algorithms (Beamforming, MUSIC, ESPRIT)
- Advanced/learning-based DOA algorithms (HyperDOA)
- Robust DOA algorithms for multipath environments
- Joint range-DOA estimation
- DOA performance metrics
"""

from .robust_doa import (
    SpatialSmoothing,
    RootMUSIC,
    BeamspaceMUSIC,
    MUSICWithRefinement,
    multipath_robust_music,
    esprit_with_spatial_smoothing,
    estimate_num_sources_aic,
    estimate_num_sources_md,
)
from .joint_estimator import (
    JointRangeAngleEstimator,
    RangeAngleSearchGrid,
    delay_music_spectrum,
    coherent_music_spectrum,
)
from .hyperdoa_wrapper import HyperDOAEstimator, HyperDOAConfig, sanity_check_hyperdoa_sign

__all__ = [
    # Robust DOA for multipath
    "SpatialSmoothing",
    "RootMUSIC",
    "BeamspaceMUSIC",
    "MUSICWithRefinement",
    "multipath_robust_music",
    "esprit_with_spatial_smoothing",
    "estimate_num_sources_aic",
    "estimate_num_sources_md",
    # Joint estimation
    "JointRangeAngleEstimator",
    "RangeAngleSearchGrid",
    "delay_music_spectrum",
    "coherent_music_spectrum",
    # HyperDOA (learning-based)
    "HyperDOAEstimator",
    "HyperDOAConfig",
    "sanity_check_hyperdoa_sign",
]