"""Mask dataclass and utilities."""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Mask:
    """Represents a segmentation mask from SAM.

    Attributes:
        data: Binary mask array of shape (H, W).
        score: Predicted IoU score from SAM.
        stability: Stability score (interior ratio).
        point: Source point that generated this mask (x, y).
        label_id: Assigned label ID (set during processing).
    """
    data: np.ndarray
    score: float
    stability: float
    point: Optional[np.ndarray] = None
    label_id: Optional[int] = None

    @property
    def area(self) -> int:
        """Number of pixels in the mask."""
        return int(self.data.sum())

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Bounding box as (x_min, y_min, x_max, y_max)."""
        rows = np.any(self.data, axis=1)
        cols = np.any(self.data, axis=0)

        if not rows.any() or not cols.any():
            return (0, 0, 0, 0)

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        return (int(x_min), int(y_min), int(x_max), int(y_max))

    def overlaps_with(self, other: "Mask", threshold: float = 0.5) -> bool:
        """Check if this mask overlaps significantly with another.

        Args:
            other: Another mask to compare.
            threshold: Minimum overlap ratio (IoU) to consider overlapping.

        Returns:
            True if masks overlap above threshold.
        """
        intersection = np.logical_and(self.data, other.data).sum()
        union = np.logical_or(self.data, other.data).sum()

        if union == 0:
            return False

        iou = intersection / union
        return iou >= threshold


def apply_black_mask(
    image: np.ndarray,
    mask: np.ndarray,
    copy: bool = True
) -> np.ndarray:
    """Apply black mask to image (set masked pixels to 0).

    This helps SAM focus on unsegmented areas by masking
    already-segmented regions with black.

    Args:
        image: RGB image array of shape (H, W, 3).
        mask: Binary mask where True/non-zero = already segmented.
        copy: If True, return a copy. If False, modify in place.

    Returns:
        Image with masked areas set to black.
    """
    if copy:
        image = image.copy()

    # Convert mask to boolean if needed
    if mask.dtype != bool:
        mask = mask.astype(bool)

    # Set masked pixels to black (0)
    image[mask] = 0

    return image


def combine_masks(masks: list[Mask]) -> np.ndarray:
    """Combine multiple masks into a single binary mask.

    Args:
        masks: List of Mask objects.

    Returns:
        Combined binary mask (union of all masks).
    """
    if not masks:
        raise ValueError("Cannot combine empty mask list")

    result = np.zeros_like(masks[0].data, dtype=bool)

    for mask in masks:
        result |= mask.data.astype(bool)

    return result


def convert_masks_to_labels(
    masks: list[Mask],
    shape: tuple[int, int],
    start_label: int = 1
) -> np.ndarray:
    """Convert list of masks to label array.

    Later masks overwrite earlier masks at overlapping pixels.
    This provides simple last-wins conflict resolution.

    Args:
        masks: List of Mask objects.
        shape: Output shape (height, width).
        start_label: First label ID to assign.

    Returns:
        Label array where each pixel has the ID of its mask.
    """
    labels = np.zeros(shape, dtype=np.uint32)

    for i, mask in enumerate(masks):
        label_id = start_label + i
        mask.label_id = label_id
        labels[mask.data.astype(bool)] = label_id

    return labels


def filter_overlapping_masks(
    masks: list[Mask],
    iou_threshold: float = 0.5
) -> list[Mask]:
    """Remove masks that overlap significantly with higher-scored masks.

    Masks are processed in order of decreasing score. A mask is removed
    if it overlaps with any already-kept mask above the IoU threshold.

    Args:
        masks: List of Mask objects.
        iou_threshold: Maximum allowed IoU overlap.

    Returns:
        Filtered list of non-overlapping masks.
    """
    if not masks:
        return []

    # Sort by score descending
    sorted_masks = sorted(masks, key=lambda m: m.score, reverse=True)

    kept = []
    for mask in sorted_masks:
        # Check overlap with all kept masks
        overlaps = any(mask.overlaps_with(k, iou_threshold) for k in kept)

        if not overlaps:
            kept.append(mask)

    return kept
