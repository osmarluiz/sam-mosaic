"""SAM2 predictor wrapper and mask utilities."""

from sam_mosaic.sam.predictor import SAMPredictor
from sam_mosaic.sam.masks import Mask, apply_black_mask

__all__ = ["SAMPredictor", "Mask", "apply_black_mask"]
