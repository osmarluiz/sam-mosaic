"""Post-processing functions for base tile labels."""

import numpy as np
from scipy import ndimage
from typing import Tuple


def remove_small_masks(labels: np.ndarray, min_area: int = 100) -> np.ndarray:
    """Remove masks smaller than min_area pixels.

    Args:
        labels: Label array (H, W) where each pixel has an instance ID.
        min_area: Minimum area in pixels. Masks smaller than this are removed.

    Returns:
        Label array with small masks removed (set to 0).
    """
    # Count pixels per label using bincount (much faster than loop)
    max_label = labels.max()
    if max_label == 0:
        return labels.copy()

    # bincount gives count for each label index
    counts = np.bincount(labels.ravel(), minlength=max_label + 1)

    # Create mask of labels to remove (those with count < min_area)
    # Label 0 (background) should never be removed
    small_labels = np.where(counts < min_area)[0]
    small_labels = small_labels[small_labels > 0]  # Exclude background

    if len(small_labels) == 0:
        return labels.copy()

    # Create lookup table: labels to remove -> 0, others -> keep
    lut = np.arange(max_label + 1, dtype=labels.dtype)
    lut[small_labels] = 0

    return lut[labels]


def merge_enclosed_masks(
    labels: np.ndarray,
    max_enclosed_area: int = 500
) -> np.ndarray:
    """Merge small masks that are completely enclosed by larger masks.

    A mask is considered enclosed if all its boundary pixels touch only
    one other label (or background at image edges).

    Args:
        labels: Label array (H, W) where each pixel has an instance ID.
        max_enclosed_area: Maximum area for a mask to be considered for merging.

    Returns:
        Label array with enclosed small masks merged into their parent.
    """
    result = labels.copy()
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]

    for label_id in unique_labels:
        mask = labels == label_id
        area = mask.sum()

        # Only consider small masks
        if area > max_enclosed_area:
            continue

        # Dilate the mask by 1 pixel to find neighbors
        dilated = ndimage.binary_dilation(mask, iterations=1)
        boundary = dilated & ~mask

        # Get all labels that touch this mask
        neighbor_labels = np.unique(labels[boundary])
        neighbor_labels = neighbor_labels[neighbor_labels > 0]  # Exclude background
        neighbor_labels = neighbor_labels[neighbor_labels != label_id]  # Exclude self

        # If enclosed by exactly one label, merge into it
        if len(neighbor_labels) == 1:
            parent_label = neighbor_labels[0]
            result[mask] = parent_label

    return result


def _fill_edge_column(
    result: np.ndarray,
    x: int,
    direction: int,
    max_distance: int
) -> None:
    """Fill empty pixels in a column by searching in given direction.

    Args:
        result: Label array to modify in-place.
        x: Column index to fill.
        direction: -1 to search left, +1 to search right.
        max_distance: Maximum search distance.
    """
    height, width = result.shape
    empty_mask = result[:, x] == 0

    if not empty_mask.any():
        return

    # For each search distance, find labels and fill empty pixels
    for d in range(1, max_distance + 1):
        search_x = x + direction * d
        if search_x < 0 or search_x >= width:
            break

        # Get labels at search position
        search_labels = result[:, search_x]

        # Fill empty pixels where we found a label
        fill_mask = empty_mask & (search_labels > 0)
        result[fill_mask, x] = search_labels[fill_mask]

        # Update empty mask (pixels we've filled are no longer empty)
        empty_mask = empty_mask & (result[:, x] == 0)

        if not empty_mask.any():
            break


def _fill_edge_row(
    result: np.ndarray,
    y: int,
    direction: int,
    max_distance: int
) -> None:
    """Fill empty pixels in a row by searching in given direction.

    Args:
        result: Label array to modify in-place.
        y: Row index to fill.
        direction: -1 to search up, +1 to search down.
        max_distance: Maximum search distance.
    """
    height, width = result.shape
    empty_mask = result[y, :] == 0

    if not empty_mask.any():
        return

    # For each search distance, find labels and fill empty pixels
    for d in range(1, max_distance + 1):
        search_y = y + direction * d
        if search_y < 0 or search_y >= height:
            break

        # Get labels at search position
        search_labels = result[search_y, :]

        # Fill empty pixels where we found a label
        fill_mask = empty_mask & (search_labels > 0)
        result[y, fill_mask] = search_labels[fill_mask]

        # Update empty mask
        empty_mask = empty_mask & (result[y, :] == 0)

        if not empty_mask.any():
            break


def complete_edges_at_discontinuities(
    labels: np.ndarray,
    tile_size: int,
    max_distance: int = 15
) -> np.ndarray:
    """Fill empty edge pixels by looking towards tile centers.

    For each discontinuity line (where tiles meet), empty pixels at the
    edge are filled with the nearest label found by looking inward.

    Args:
        labels: Label array (H, W) where each pixel has an instance ID.
        tile_size: Size of tiles (e.g., 2000).
        max_distance: Maximum distance to search for a label (default 15).

    Returns:
        Label array with edge pixels filled.
    """
    result = labels.copy()
    height, width = labels.shape

    # Find all vertical discontinuity x-coordinates
    v_disc_xs = list(range(tile_size, width, tile_size))

    # Find all horizontal discontinuity y-coordinates
    h_disc_ys = list(range(tile_size, height, tile_size))

    # Process vertical discontinuities
    for disc_x in v_disc_xs:
        # Left side (x = disc_x - 1): search LEFT (direction = -1)
        if disc_x > 0:
            _fill_edge_column(result, disc_x - 1, direction=-1, max_distance=max_distance)

        # Right side (x = disc_x): search RIGHT (direction = +1)
        if disc_x < width:
            _fill_edge_column(result, disc_x, direction=+1, max_distance=max_distance)

    # Process horizontal discontinuities
    for disc_y in h_disc_ys:
        # Top side (y = disc_y - 1): search UP (direction = -1)
        if disc_y > 0:
            _fill_edge_row(result, disc_y - 1, direction=-1, max_distance=max_distance)

        # Bottom side (y = disc_y): search DOWN (direction = +1)
        if disc_y < height:
            _fill_edge_row(result, disc_y, direction=+1, max_distance=max_distance)

    return result


def postprocess_base_labels(
    labels: np.ndarray,
    tile_size: int,
    min_area: int = 100,
    max_enclosed_area: int = 500,
    edge_max_distance: int = 15
) -> np.ndarray:
    """Apply full post-processing pipeline to base tile labels.

    Order of operations:
    1. Remove small masks (< min_area)
    2. Merge small enclosed masks into parent
    3. Complete edges at discontinuities

    Args:
        labels: Label array (H, W) where each pixel has an instance ID.
        tile_size: Size of tiles (e.g., 2000).
        min_area: Minimum area in pixels for mask removal.
        max_enclosed_area: Maximum area for enclosed mask merging.
        edge_max_distance: Maximum distance to search for edge completion.

    Returns:
        Post-processed label array.
    """
    # Step 1: Remove small masks
    result = remove_small_masks(labels, min_area)

    # Step 2: Merge enclosed small masks
    result = merge_enclosed_masks(result, max_enclosed_area)

    # Step 3: Complete edges at discontinuities
    result = complete_edges_at_discontinuities(result, tile_size, edge_max_distance)

    return result
