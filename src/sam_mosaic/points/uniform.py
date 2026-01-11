"""Uniform grid point generation."""

import numpy as np


def make_uniform_grid(
    height: int,
    width: int,
    points_per_side: int,
    margin: int = 0
) -> np.ndarray:
    """Generate uniform grid of points.

    Creates a regular grid of points evenly distributed across
    the image dimensions.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        points_per_side: Number of points per axis (creates NxN grid).
        margin: Margin from edges in pixels.

    Returns:
        Array of shape (N*N, 2) with (x, y) coordinates.

    Example:
        >>> points = make_uniform_grid(1000, 1000, 64)
        >>> points.shape
        (4096, 2)
    """
    x_start = margin
    x_end = width - 1 - margin
    y_start = margin
    y_end = height - 1 - margin

    xs = np.linspace(x_start, x_end, points_per_side)
    ys = np.linspace(y_start, y_end, points_per_side)

    xx, yy = np.meshgrid(xs, ys)
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)

    return points.astype(np.int32)


def make_grid_in_region(
    x_start: int,
    y_start: int,
    x_end: int,
    y_end: int,
    points_per_side: int
) -> np.ndarray:
    """Generate uniform grid within a specific region.

    Args:
        x_start: Left boundary.
        y_start: Top boundary.
        x_end: Right boundary.
        y_end: Bottom boundary.
        points_per_side: Number of points per axis.

    Returns:
        Array of shape (N*N, 2) with (x, y) coordinates.
    """
    xs = np.linspace(x_start, x_end, points_per_side)
    ys = np.linspace(y_start, y_end, points_per_side)

    xx, yy = np.meshgrid(xs, ys)
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)

    return points.astype(np.int32)
