"""Label merging and post-processing."""

from sam_mosaic.merge.discontinuity import merge_at_discontinuities
from sam_mosaic.merge.enclosed import merge_enclosed_regions
from sam_mosaic.merge.small import remove_small_regions

__all__ = [
    "merge_at_discontinuities",
    "merge_enclosed_regions",
    "remove_small_regions",
]
