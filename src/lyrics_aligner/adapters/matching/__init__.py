"""Feature matching adapters."""

from lyrics_aligner.adapters.matching.nearest_neighbor import (
    NearestNeighborFeatureMatcher,
    NearestNeighborFeatureMatcherConfig,
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
    "NearestNeighborFeatureMatcher",
    "NearestNeighborFeatureMatcherConfig",
    "StabilizedFeatureMatcher",
    "StabilizedFeatureMatcherConfig",
    "TrackingFeatureMatcher",
    "TrackingFeatureMatcherConfig",
]
