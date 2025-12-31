"""Multi-band mask merging with priority-based NMS."""

from typing import List, Tuple, Dict, Optional
import numpy as np

from sam_mosaic.segmentation.sam import Mask


def merge_bands(
    bands: Dict[int, np.ndarray],
    priority: List[int] = [2, 1, 0]
) -> np.ndarray:
    """Merge multiple label bands with priority ordering.

    Higher priority bands overwrite lower priority at each pixel.
    Default priority: corners (2) > V/H (1) > base (0).

    Args:
        bands: Dictionary mapping band ID to label array (H, W).
        priority: List of band IDs in descending priority order.

    Returns:
        Merged label array.
    """
    if not bands:
        raise ValueError("At least one band required")

    # Get shape from first band
    first_band = next(iter(bands.values()))
    height, width = first_band.shape
    merged = np.zeros((height, width), dtype=np.uint32)

    # Apply in reverse priority (lowest first, highest last)
    for band_id in reversed(priority):
        if band_id not in bands:
            continue
        band = bands[band_id]
        # Only overwrite where band has labels
        merged[band > 0] = band[band > 0]

    return merged


def merge_bands_relabel(
    bands: Dict[int, np.ndarray],
    priority: List[int] = [2, 1, 0]
) -> Tuple[np.ndarray, int]:
    """Merge bands with sequential relabeling.

    Assigns globally unique IDs to all instances across bands.

    Args:
        bands: Dictionary mapping band ID to label array.
        priority: List of band IDs in descending priority order.

    Returns:
        Tuple of (merged labels, total instance count).
    """
    if not bands:
        raise ValueError("At least one band required")

    first_band = next(iter(bands.values()))
    height, width = first_band.shape
    merged = np.zeros((height, width), dtype=np.uint32)

    current_id = 1

    # Process in priority order (highest first)
    for band_id in priority:
        if band_id not in bands:
            continue

        band = bands[band_id]
        unique_labels = np.unique(band[band > 0])

        for old_label in unique_labels:
            mask = band == old_label
            # Only assign if not already covered by higher priority
            uncovered = mask & (merged == 0)
            if uncovered.any():
                merged[uncovered] = current_id
                current_id += 1

    return merged, current_id - 1


def apply_nms(
    masks: List[Mask],
    iou_thresh: float = 0.5
) -> List[Mask]:
    """Apply non-maximum suppression to mask list.

    Removes highly overlapping masks, keeping higher-scored ones.

    Args:
        masks: List of Mask objects sorted by score (descending).
        iou_thresh: IoU threshold for suppression.

    Returns:
        Filtered list of masks.
    """
    if len(masks) <= 1:
        return masks

    # Sort by score descending
    masks = sorted(masks, key=lambda m: m.score, reverse=True)

    keep = []
    suppressed = set()

    for i, mask_i in enumerate(masks):
        if i in suppressed:
            continue

        keep.append(mask_i)

        for j in range(i + 1, len(masks)):
            if j in suppressed:
                continue

            iou = _compute_iou(mask_i.mask, masks[j].mask)
            if iou > iou_thresh:
                suppressed.add(j)

    return keep


def _compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    intersection = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()

    if union == 0:
        return 0.0
    return intersection / union


def masks_to_band(
    masks: List[Mask],
    height: int,
    width: int,
    offset_x: int = 0,
    offset_y: int = 0
) -> np.ndarray:
    """Convert mask list to label band at global coordinates.

    Args:
        masks: List of Mask objects (in local tile coordinates).
        height: Output image height.
        width: Output image width.
        offset_x: X-offset of tile in global coordinates.
        offset_y: Y-offset of tile in global coordinates.

    Returns:
        Label array at global coordinates.
    """
    labels = np.zeros((height, width), dtype=np.uint32)

    for i, mask in enumerate(masks, start=1):
        # Get local mask bounds
        h, w = mask.mask.shape

        # Compute global bounds
        y_start = offset_y
        y_end = min(offset_y + h, height)
        x_start = offset_x
        x_end = min(offset_x + w, width)

        # Compute local bounds for mask
        local_y_end = y_end - offset_y
        local_x_end = x_end - offset_x

        # Apply mask
        local_mask = mask.mask[:local_y_end, :local_x_end]
        labels[y_start:y_end, x_start:x_end][local_mask > 0] = i

    return labels


def stitch_tile_labels(
    tiles: List[Tuple[np.ndarray, int, int]],
    height: int,
    width: int
) -> np.ndarray:
    """Stitch multiple tile label arrays into full image.

    Args:
        tiles: List of (labels, x_offset, y_offset) tuples.
        height: Output image height.
        width: Output image width.

    Returns:
        Stitched label array with unique IDs.
    """
    full = np.zeros((height, width), dtype=np.uint32)
    current_max = 0

    for labels, x_off, y_off in tiles:
        h, w = labels.shape

        # Offset labels to maintain uniqueness
        tile_labels = labels.copy()
        tile_labels[tile_labels > 0] += current_max

        # Get bounds
        y_end = min(y_off + h, height)
        x_end = min(x_off + w, width)
        local_h = y_end - y_off
        local_w = x_end - x_off

        # Stitch
        target = full[y_off:y_end, x_off:x_end]
        source = tile_labels[:local_h, :local_w]

        # Only fill where target is empty
        empty = target == 0
        target[empty] = source[empty]

        current_max = full.max()

    return full
