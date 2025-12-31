"""Point grid generation for SAM prompts."""

from typing import Tuple, Optional
import numpy as np
from scipy import ndimage


def make_uniform_grid(
    height: int,
    width: int,
    points_per_side: int
) -> np.ndarray:
    """Generate uniform grid of points.

    Args:
        height: Image/tile height.
        width: Image/tile width.
        points_per_side: Number of points per side (creates NxN grid).

    Returns:
        Array of shape (N*N, 2) with (x, y) coordinates.
    """
    xs = np.linspace(0, width - 1, points_per_side)
    ys = np.linspace(0, height - 1, points_per_side)

    # Create meshgrid
    xx, yy = np.meshgrid(xs, ys)
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)

    return points.astype(np.int32)


def make_v_grid(
    tile_size: int,
    zone_width: int = 100,
    n_across: int = 5,
    n_along: int = 100
) -> np.ndarray:
    """Generate fixed grid for vertical discontinuity tiles.

    Points are in a vertical band centered at x = tile_size/2.
    Always the same relative positions regardless of tile location.

    Args:
        tile_size: Tile size (e.g., 2000). Discontinuity is at tile_size/2.
        zone_width: Width of point zone (default 100 = 50px each side).
        n_across: Number of columns (default 5).
        n_along: Number of rows (default 100).

    Returns:
        Array of shape (n_across * n_along, 2) with (x, y) coordinates.

    Example for tile_size=2000, zone_width=100:
        Discontinuity at x=1000
        Points from x=950 to x=1050 (5 columns)
        Points from y=10 to y=1990 (100 rows)
    """
    center_x = tile_size // 2
    half_zone = zone_width // 2

    xs = np.linspace(center_x - half_zone, center_x + half_zone, n_across)
    ys = np.linspace(10, tile_size - 10, n_along)  # 10px margin

    xx, yy = np.meshgrid(xs, ys)
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)

    return points.astype(np.int32)


def make_h_grid(
    tile_size: int,
    zone_width: int = 100,
    n_across: int = 5,
    n_along: int = 100
) -> np.ndarray:
    """Generate fixed grid for horizontal discontinuity tiles.

    Points are in a horizontal band centered at y = tile_size/2.
    Always the same relative positions regardless of tile location.

    Args:
        tile_size: Tile size (e.g., 2000). Discontinuity is at tile_size/2.
        zone_width: Height of point zone (default 100 = 50px each side).
        n_across: Number of rows perpendicular to line (default 5).
        n_along: Number of columns along the line (default 100).

    Returns:
        Array of shape (n_across * n_along, 2) with (x, y) coordinates.

    Example for tile_size=2000, zone_width=100:
        Discontinuity at y=1000
        Points from y=950 to y=1050 (5 rows)
        Points from x=10 to x=1990 (100 columns)
    """
    center_y = tile_size // 2
    half_zone = zone_width // 2

    xs = np.linspace(10, tile_size - 10, n_along)  # 10px margin
    ys = np.linspace(center_y - half_zone, center_y + half_zone, n_across)

    xx, yy = np.meshgrid(xs, ys)
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)

    return points.astype(np.int32)


def make_corner_grid(
    tile_size: int,
    zone_width: int = 100,
    n_points: int = 5
) -> np.ndarray:
    """Generate fixed grid for corner tiles.

    Points are in a square zone centered at (tile_size/2, tile_size/2).
    Always the same relative positions regardless of tile location.

    Args:
        tile_size: Tile size (e.g., 2000). Corner is at (tile_size/2, tile_size/2).
        zone_width: Size of point zone (default 100 = 50px each direction).
        n_points: Points per axis (default 5 = 25 total).

    Returns:
        Array of shape (n_points * n_points, 2) with (x, y) coordinates.

    Example for tile_size=2000, zone_width=100:
        Corner at (1000, 1000)
        Points from (950,950) to (1050,1050) in 5x5 grid
    """
    center = tile_size // 2
    half_zone = zone_width // 2

    xs = np.linspace(center - half_zone, center + half_zone, n_points)
    ys = np.linspace(center - half_zone, center + half_zone, n_points)

    xx, yy = np.meshgrid(xs, ys)
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)

    return points.astype(np.int32)


def filter_by_mask(
    points: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """Filter points that fall on already-segmented regions.

    Points on non-zero mask pixels are removed.

    Args:
        points: Array of shape (N, 2) with (x, y) coordinates.
        mask: Binary mask array (H, W) where non-zero = already segmented.

    Returns:
        Filtered points array.
    """
    valid = []
    for x, y in points:
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            if mask[int(y), int(x)] == 0:
                valid.append([x, y])

    if len(valid) == 0:
        return np.array([], dtype=np.int32).reshape(0, 2)

    return np.array(valid, dtype=np.int32)


def add_offset(
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
    return points + np.array([[offset_x, offset_y]])


def erode_mask(mask: np.ndarray, iterations: int = 5) -> np.ndarray:
    """Erode binary mask to avoid edges.

    Args:
        mask: Binary mask (H, W) where True = valid area.
        iterations: Number of erosion iterations (pixels).

    Returns:
        Eroded mask.
    """
    if iterations <= 0:
        return mask
    return ndimage.binary_erosion(mask, iterations=iterations)


def make_kmeans_points(
    valid_mask: np.ndarray,
    n_points: int = 64,
    erosion: int = 5,
    max_samples: int = 10000
) -> np.ndarray:
    """Generate well-distributed points using K-means clustering.

    Points are placed in valid (unmasked) areas, eroded to avoid edges.

    Args:
        valid_mask: Boolean mask (H, W) where True = valid for points.
        n_points: Number of points to generate.
        erosion: Erosion iterations to avoid placing points near edges.
        max_samples: Max samples for K-means fitting.

    Returns:
        Array of shape (N, 2) with (x, y) coordinates.
    """
    # Apply erosion
    eroded = erode_mask(valid_mask, erosion)

    # Get valid coordinates (y, x format from numpy)
    valid_coords = np.argwhere(eroded)

    if len(valid_coords) == 0:
        return np.array([], dtype=np.int32).reshape(0, 2)

    if len(valid_coords) <= n_points:
        # Not enough points, return all
        # Convert from (y, x) to (x, y)
        return valid_coords[:, ::-1].astype(np.int32)

    # Subsample for K-means if too many
    if len(valid_coords) > max_samples:
        idx = np.random.choice(len(valid_coords), max_samples, replace=False)
        coords_sample = valid_coords[idx]
    else:
        coords_sample = valid_coords

    # Run K-means
    try:
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(
            n_clusters=n_points,
            random_state=42,
            batch_size=1000,
            n_init=3
        )
        kmeans.fit(coords_sample)

        # Find closest valid point to each centroid
        points = []
        for cy, cx in kmeans.cluster_centers_:
            distances = np.sqrt(
                (coords_sample[:, 0] - cy)**2 +
                (coords_sample[:, 1] - cx)**2
            )
            closest_idx = distances.argmin()
            # Convert (y, x) to (x, y)
            y, x = coords_sample[closest_idx]
            points.append([x, y])

        return np.array(points, dtype=np.int32)

    except ImportError:
        # Fallback: random sampling if sklearn not available
        idx = np.random.choice(len(valid_coords), n_points, replace=False)
        # Convert (y, x) to (x, y)
        return valid_coords[idx, ::-1].astype(np.int32)
