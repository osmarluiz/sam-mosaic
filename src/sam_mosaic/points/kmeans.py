"""K-means based point generation for residual areas."""

import numpy as np
from scipy import ndimage
from typing import Optional


def _sample_valid_coords(
    mask: np.ndarray,
    n_samples: int,
    rng: np.random.Generator
) -> np.ndarray:
    """Sample coordinates from valid (True) positions using rejection sampling.

    This is memory-efficient: instead of creating an array with ALL valid
    coordinates (which can be huge for large images), we randomly sample
    coordinates and keep only those that fall on valid pixels.

    Args:
        mask: Boolean mask (H, W) where True = valid for sampling.
        n_samples: Number of samples to collect.
        rng: NumPy random generator for reproducibility.

    Returns:
        Array of shape (N, 2) with (y, x) coordinates (numpy convention).
        N may be less than n_samples if valid area is very sparse.
    """
    h, w = mask.shape
    n_valid = mask.sum()

    if n_valid == 0:
        return np.array([], dtype=np.int32).reshape(0, 2)

    if n_valid <= n_samples:
        # Few valid pixels - use argwhere (small memory footprint)
        return np.argwhere(mask)

    # Estimate acceptance rate and batch size
    acceptance_rate = n_valid / mask.size
    # Oversample to reduce iterations (minimum 2x, up to 10x for sparse masks)
    oversample_factor = min(10, max(2, int(1 / acceptance_rate) + 1))
    batch_size = min(n_samples * oversample_factor, 100000)

    samples = []
    max_iterations = 100  # Safety limit

    for _ in range(max_iterations):
        # Generate random coordinates
        ys = rng.integers(0, h, size=batch_size)
        xs = rng.integers(0, w, size=batch_size)

        # Keep only valid ones
        valid = mask[ys, xs]
        valid_ys = ys[valid]
        valid_xs = xs[valid]

        # Add to samples
        for y, x in zip(valid_ys, valid_xs):
            samples.append([y, x])
            if len(samples) >= n_samples:
                return np.array(samples[:n_samples], dtype=np.int64)

        # If we collected enough, return
        if len(samples) >= n_samples:
            break

    # Return what we have (might be less than n_samples for very sparse masks)
    return np.array(samples, dtype=np.int64) if samples else np.array([], dtype=np.int64).reshape(0, 2)


def make_kmeans_points(
    valid_mask: np.ndarray,
    n_points: int = 64,
    erosion_iterations: int = 5,
    max_samples: int = 10000,
    seed: Optional[int] = 42
) -> np.ndarray:
    """Generate well-distributed points using K-means clustering.

    Points are placed in valid (unmasked) areas using K-means to ensure
    good spatial distribution. The mask is eroded to avoid placing
    points too close to edges.

    Uses memory-efficient rejection sampling instead of np.argwhere to
    avoid allocating huge coordinate arrays for large images.

    Args:
        valid_mask: Boolean mask (H, W) where True = valid for points.
        n_points: Number of points to generate.
        erosion_iterations: Iterations to erode mask (avoid edges).
        max_samples: Maximum samples for K-means fitting.
        seed: Random seed for reproducibility. None for random behavior.

    Returns:
        Array of shape (N, 2) with (x, y) coordinates.
        N may be less than n_points if not enough valid area.

    Example:
        >>> mask = np.zeros((1000, 1000), dtype=bool)
        >>> mask[100:900, 100:900] = True  # Valid region
        >>> points = make_kmeans_points(mask, n_points=32)
    """
    # Create random generator for reproducibility
    rng = np.random.default_rng(seed)

    # Apply erosion to avoid edge points
    if erosion_iterations > 0:
        eroded = ndimage.binary_erosion(valid_mask, iterations=erosion_iterations)
    else:
        eroded = valid_mask

    # Count valid pixels (cheap operation)
    n_valid = eroded.sum()

    if n_valid == 0:
        return np.array([], dtype=np.int32).reshape(0, 2)

    if n_valid <= n_points:
        # Few valid pixels - get all of them
        valid_coords = np.argwhere(eroded)
        return valid_coords[:, ::-1].astype(np.int32)  # Convert to (x, y)

    # Sample coordinates efficiently (memory-efficient for large images)
    coords_sample = _sample_valid_coords(eroded, max_samples, rng)

    if len(coords_sample) == 0:
        return np.array([], dtype=np.int32).reshape(0, 2)

    if len(coords_sample) <= n_points:
        return coords_sample[:, ::-1].astype(np.int32)

    # Run K-means clustering
    try:
        from sklearn.cluster import MiniBatchKMeans

        kmeans = MiniBatchKMeans(
            n_clusters=n_points,
            random_state=seed if seed is not None else 42,
            batch_size=min(1000, len(coords_sample)),
            n_init=3
        )
        kmeans.fit(coords_sample)

        # Find closest valid point to each centroid (vectorized)
        centers = kmeans.cluster_centers_  # Shape: (n_points, 2)
        points = []

        for cy, cx in centers:
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
        indices = rng.choice(len(coords_sample), n_points, replace=False)
        return coords_sample[indices, ::-1].astype(np.int32)


def make_residual_points(
    combined_mask: np.ndarray,
    n_points: int = 64,
    erosion_iterations: int = 5,
    seed: Optional[int] = 42
) -> np.ndarray:
    """Generate K-means points in unsegmented (residual) areas.

    This is the main function for multi-pass segmentation, placing
    points in areas that haven't been segmented yet.

    Args:
        combined_mask: Binary mask where non-zero = already segmented.
        n_points: Number of points to generate.
        erosion_iterations: Iterations to erode valid area.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (N, 2) with (x, y) coordinates.
    """
    # Valid area = NOT already segmented
    valid_mask = combined_mask == 0

    return make_kmeans_points(
        valid_mask,
        n_points=n_points,
        erosion_iterations=erosion_iterations,
        seed=seed
    )
