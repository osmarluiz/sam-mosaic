"""Dense grid point generation for urban/small object segmentation."""

import numpy as np
from scipy import ndimage

from sam_mosaic.points.uniform import make_uniform_grid
from sam_mosaic.points.filter import filter_points_by_mask


def make_dense_grid_points(
    combined_mask: np.ndarray,
    points_per_side: int = 64,
    erosion_iterations: int = 5
) -> np.ndarray:
    """Generate uniform grid points, filtered by already-segmented areas.

    This strategy uses a fixed dense grid for ALL passes, simply filtering
    out points that fall on already-segmented regions. Unlike K-means which
    clusters points (potentially missing isolated small objects), this ensures
    uniform coverage across all unsegmented areas.

    Best for:
    - Urban imagery with many small objects (cars, buildings)
    - Scenes with scattered isolated objects
    - When uniform coverage is more important than adaptive clustering

    Args:
        combined_mask: Binary mask (H, W) where non-zero = already segmented.
        points_per_side: Grid density (creates NxN grid).
        erosion_iterations: Erode the valid area before filtering.
            Smaller values (e.g., 5) allow points closer to segment edges,
            good for capturing small objects near existing segments.

    Returns:
        Array of shape (N, 2) with (x, y) coordinates of valid points.
        Points in already-segmented areas are excluded.

    Example:
        >>> mask = np.zeros((1000, 1000), dtype=np.uint8)
        >>> mask[100:500, 100:500] = 1  # Already segmented
        >>> points = make_dense_grid_points(mask, points_per_side=64)
        >>> # Points will cover the unsegmented areas uniformly
    """
    height, width = combined_mask.shape

    # Generate full uniform grid
    all_points = make_uniform_grid(height, width, points_per_side)

    if len(all_points) == 0:
        return np.array([], dtype=np.int32).reshape(0, 2)

    # Create valid mask (areas NOT yet segmented)
    valid_mask = combined_mask == 0

    # Apply erosion to avoid placing points too close to edges
    if erosion_iterations > 0:
        valid_mask = ndimage.binary_erosion(valid_mask, iterations=erosion_iterations)

    # Convert to uint8 for filter function (expects 0 = valid, non-zero = invalid)
    # Our valid_mask is True = valid, so we invert it
    filter_mask = (~valid_mask).astype(np.uint8)

    # Filter points - keep only those on valid (unsegmented) areas
    filtered_points = filter_points_by_mask(all_points, filter_mask)

    return filtered_points


def make_dense_grid_residual(
    combined_mask: np.ndarray,
    height: int,
    width: int,
    points_per_side: int = 64,
    erosion_iterations: int = 5
) -> np.ndarray:
    """Generate dense grid points for residual (unsegmented) areas.

    Convenience wrapper that matches the signature pattern of make_kmeans_points.

    Args:
        combined_mask: Binary mask where non-zero = already segmented.
        height: Image height (for grid generation).
        width: Image width (for grid generation).
        points_per_side: Grid density.
        erosion_iterations: Erosion iterations for valid area.

    Returns:
        Array of shape (N, 2) with (x, y) coordinates.
    """
    return make_dense_grid_points(
        combined_mask,
        points_per_side=points_per_side,
        erosion_iterations=erosion_iterations
    )
