"""Point filtering utilities."""

import numpy as np


def filter_points_by_mask(
    points: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """Filter out points that fall on already-segmented regions.

    Uses vectorized indexing for O(n) performance.

    Args:
        points: Array of shape (N, 2) with (x, y) coordinates.
        mask: Binary mask (H, W) where non-zero = already segmented.

    Returns:
        Filtered points array with only points on unsegmented areas.

    Example:
        >>> points = np.array([[10, 10], [50, 50], [100, 100]])
        >>> mask = np.zeros((200, 200), dtype=np.uint8)
        >>> mask[40:60, 40:60] = 1  # Mark (50, 50) as segmented
        >>> filtered = filter_points_by_mask(points, mask)
        >>> len(filtered)
        2
    """
    if len(points) == 0:
        return np.array([], dtype=np.int32).reshape(0, 2)

    # Extract coordinates
    xs = points[:, 0].astype(np.int32)
    ys = points[:, 1].astype(np.int32)

    # Check bounds (vectorized)
    in_bounds = (
        (ys >= 0) & (ys < mask.shape[0]) &
        (xs >= 0) & (xs < mask.shape[1])
    )

    # Check mask values for in-bounds points (vectorized lookup)
    valid = np.zeros(len(points), dtype=bool)
    valid[in_bounds] = mask[ys[in_bounds], xs[in_bounds]] == 0

    filtered = points[valid]

    if len(filtered) == 0:
        return np.array([], dtype=np.int32).reshape(0, 2)

    return filtered.astype(np.int32)


def filter_points_in_region(
    points: np.ndarray,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int
) -> np.ndarray:
    """Filter points to only those within a bounding box.

    Args:
        points: Array of shape (N, 2) with (x, y) coordinates.
        x_min: Minimum x coordinate.
        y_min: Minimum y coordinate.
        x_max: Maximum x coordinate.
        y_max: Maximum y coordinate.

    Returns:
        Filtered points within the bounding box.
    """
    if len(points) == 0:
        return np.array([], dtype=np.int32).reshape(0, 2)

    xs = points[:, 0]
    ys = points[:, 1]

    valid = (
        (xs >= x_min) & (xs <= x_max) &
        (ys >= y_min) & (ys <= y_max)
    )

    return points[valid].astype(np.int32)


def add_offset_to_points(
    points: np.ndarray,
    offset_x: int,
    offset_y: int
) -> np.ndarray:
    """Add offset to convert local tile coords to global image coords.

    Args:
        points: Array of shape (N, 2) with (x, y) coordinates.
        offset_x: X offset to add.
        offset_y: Y offset to add.

    Returns:
        Points with offset applied.
    """
    if len(points) == 0:
        return points

    return points + np.array([[offset_x, offset_y]], dtype=np.int32)
