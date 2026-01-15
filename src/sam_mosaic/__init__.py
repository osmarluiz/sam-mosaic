"""
SAM-Mosaic: Large-scale image segmentation using SAM2.

This package provides tools for segmenting very large images (e.g., satellite imagery)
using the Segment Anything Model 2 (SAM2) with intelligent tiling and merging.

Key features:
- Multi-pass segmentation with adaptive thresholds
- Black mask focusing for iterative refinement
- Optimized merge at tile boundaries (O(n) with LUT)
- K-means point placement in residual areas
- YAML configuration with CLI overrides for ablation studies

Example usage:
    from sam_mosaic import segment_image, segment_with_params

    # Simple usage with config file
    result = segment_image(
        "input.tif",
        "output/",
        checkpoint="path/to/sam2.pt"
    )

    # Direct parameter control for ablations
    result = segment_with_params(
        "input.tif",
        "output/",
        checkpoint="path/to/sam2.pt",
        iou_start=0.86,
        use_black_mask=False,
        max_passes=1
    )
"""

# =============================================================================
# Environment setup (must be before other imports)
# =============================================================================
import os
import sys
import warnings

# Silence SAM2 optional CUDA extension warning (doesn't affect results)
warnings.filterwarnings("ignore", message="cannot import name '_C' from 'sam2'")

# Fix MiniBatchKMeans memory leak on Windows with MKL
# See: https://github.com/scikit-learn/scikit-learn/issues/23246
if sys.platform == "win32" and "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

# =============================================================================

__version__ = "2.0.0"
__author__ = "Osmar Luiz Ferreira de Carvalho"

from sam_mosaic.api import segment_image, segment_with_params
from sam_mosaic.config import Config, load_config
from sam_mosaic.core.pipeline import Pipeline

__all__ = [
    "segment_image",
    "segment_with_params",
    "Config",
    "load_config",
    "Pipeline",
    "__version__",
]
