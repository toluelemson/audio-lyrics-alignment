"""Feature matching adapters."""

from lyrics_aligner.adapters.matching.nearest_neighbor import (
    NearestNeighborFeatureMatcher,
    NearestNeighborFeatureMatcherConfig,
)
from lyrics_aligner.adapters.matching.stabilized import (
    StabilizedFeatureMatcher,
    StabilizedFeatureMatcherConfig,
)

__all__ = [
    "NearestNeighborFeatureMatcher",
    "NearestNeighborFeatureMatcherConfig",
    "StabilizedFeatureMatcher",
    "StabilizedFeatureMatcherConfig",
]
