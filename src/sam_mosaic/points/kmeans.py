"""K-means based point generation for residual areas."""

import numpy as np
from scipy import ndimage


def make_kmeans_points(
    valid_mask: np.ndarray,
    n_points: int = 64,
    erosion_iterations: int = 5,
    max_samples: int = 10000
) -> np.ndarray:
    """Generate well-distributed points using K-means clustering.

    Points are placed in valid (unmasked) areas using K-means to ensure
    good spatial distribution. The mask is eroded to avoid placing
    points too close to edges.

    Args:
        valid_mask: Boolean mask (H, W) where True = valid for points.
        n_points: Number of points to generate.
        erosion_iterations: Iterations to erode mask (avoid edges).
        max_samples: Maximum samples for K-means fitting.

    Returns:
        Array of shape (N, 2) with (x, y) coordinates.
        N may be less than n_points if not enough valid area.

    Example:
        >>> mask = np.zeros((1000, 1000), dtype=bool)
        >>> mask[100:900, 100:900] = True  # Valid region
        >>> points = make_kmeans_points(mask, n_points=32)
    """
    # Apply erosion to avoid edge points
    if erosion_iterations > 0:
        eroded = ndimage.binary_erosion(valid_mask, iterations=erosion_iterations)
    else:
        eroded = valid_mask

    # Get valid coordinates (y, x format from numpy)
    valid_coords = np.argwhere(eroded)

    if len(valid_coords) == 0:
        return np.array([], dtype=np.int32).reshape(0, 2)

    if len(valid_coords) <= n_points:
        # Not enough points, return all (converted to x, y)
        return valid_coords[:, ::-1].astype(np.int32)

    # Subsample for K-means if too many points
    if len(valid_coords) > max_samples:
        indices = np.random.choice(len(valid_coords), max_samples, replace=False)
        coords_sample = valid_coords[indices]
    else:
        coords_sample = valid_coords

    # Run K-means clustering
    try:
        from sklearn.cluster import MiniBatchKMeans

        kmeans = MiniBatchKMeans(
            n_clusters=n_points,
            random_state=42,
            batch_size=min(1000, len(coords_sample)),
            n_init=3
        )
        kmeans.fit(coords_sample)

        # Find closest valid point to each centroid
        points = []
        for cy, cx in kmeans.cluster_centers_:
            distances = np.sqrt(
                (coords_sample[:, 0] - cy) ** 2 +
                (coords_sample[:, 1] - cx) ** 2
            )
            closest_idx = distances.argmin()
            y, x = coords_sample[closest_idx]
            points.append([x, y])  # Convert to (x, y)

        return np.array(points, dtype=np.int32)

    except ImportError:
        # Fallback: random sampling if sklearn not available
        indices = np.random.choice(len(valid_coords), n_points, replace=False)
        return valid_coords[indices, ::-1].astype(np.int32)


def make_residual_points(
    combined_mask: np.ndarray,
    n_points: int = 64,
    erosion_iterations: int = 5
) -> np.ndarray:
    """Generate K-means points in unsegmented (residual) areas.

    This is the main function for multi-pass segmentation, placing
    points in areas that haven't been segmented yet.

    Args:
        combined_mask: Binary mask where non-zero = already segmented.
        n_points: Number of points to generate.
        erosion_iterations: Iterations to erode valid area.

    Returns:
        Array of shape (N, 2) with (x, y) coordinates.
    """
    # Valid area = NOT already segmented
    valid_mask = combined_mask == 0

    return make_kmeans_points(
        valid_mask,
        n_points=n_points,
        erosion_iterations=erosion_iterations
    )
