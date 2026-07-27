"""Feature matching adapters."""

from lyrics_aligner.adapters.matching.coarse_fingerprint import (
    CoarseAnchorFeatureMatcher,
    CoarseFingerprintMatcher,
    CoarseFingerprintMatcherConfig,
)
from lyrics_aligner.adapters.matching.nearest_neighbor import (
    NearestNeighborFeatureMatcher,
    NearestNeighborFeatureMatcherConfig,
)
from lyrics_aligner.adapters.matching.rolling_window import (
    RollingWindowFeatureMatcher,
    RollingWindowFeatureMatcherConfig,
)
from lyrics_aligner.adapters.matching.stabilized import (
    StabilizedFeatureMatcher,
    StabilizedFeatureMatcherConfig,
)
from lyrics_aligner.adapters.matching.tracking import (
    TrackingFeatureMatcher,
    TrackingFeatureMatcherConfig,
)

__all__ = [
    "CoarseAnchorFeatureMatcher",
    "CoarseFingerprintMatcher",
    "CoarseFingerprintMatcherConfig",
    "NearestNeighborFeatureMatcher",
    "NearestNeighborFeatureMatcherConfig",
    "RollingWindowFeatureMatcher",
    "RollingWindowFeatureMatcherConfig",
    "StabilizedFeatureMatcher",
    "StabilizedFeatureMatcherConfig",
    "TrackingFeatureMatcher",
    "TrackingFeatureMatcherConfig",
]
