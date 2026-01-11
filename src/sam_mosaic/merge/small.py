"""Remove small regions from label arrays."""

import numpy as np
from scipy import ndimage


def remove_small_regions(
    labels: np.ndarray,
    min_area: int = 100
) -> np.ndarray:
    """Remove regions smaller than minimum area.

    Uses connected component analysis to identify individual fragments,
    so disconnected fragments of the same label are evaluated separately.
    This matches the original exp_plant23 behavior.

    Args:
        labels: Label array of shape (H, W).
        min_area: Minimum area in pixels to keep a region.

    Returns:
        Label array with small fragments removed (set to 0).

    Example:
        >>> labels = np.array([[1, 1, 2], [1, 1, 0], [0, 0, 0]])
        >>> cleaned = remove_small_regions(labels, min_area=3)
        >>> # Region 2 (area=1) is removed, region 1 (area=4) is kept
    """
    if labels.max() == 0:
        return labels.copy()

    # Find connected components (each fragment gets unique ID)
    # This separates disconnected fragments of the same label
    structure = ndimage.generate_binary_structure(2, 1)  # 4-connectivity
    components, n_components = ndimage.label(labels > 0, structure=structure)

    if n_components == 0:
        return labels.copy()

    # Count pixels per component
    component_sizes = np.bincount(components.ravel(), minlength=n_components + 1)

    # Find small components (< min_area)
    small_components = np.where(component_sizes < min_area)[0]
    small_components = small_components[small_components > 0]  # Exclude background

    if len(small_components) == 0:
        return labels.copy()

    # Create mask of pixels to remove
    remove_mask = np.isin(components, small_components)

    # Copy labels and set small fragments to 0
    result = labels.copy()
    result[remove_mask] = 0

    return result


def remove_small_regions_by_ratio(
    labels: np.ndarray,
    reference_area: int,
    min_ratio: float = 0.01
) -> np.ndarray:
    """Remove regions smaller than a fraction of reference area.

    Useful when the appropriate minimum area depends on image/tile size.

    Args:
        labels: Label array of shape (H, W).
        reference_area: Reference area (e.g., tile area).
        min_ratio: Minimum ratio of reference area to keep.

    Returns:
        Label array with small regions removed.
    """
    min_area = int(reference_area * min_ratio)
    return remove_small_regions(labels, min_area)


def get_region_areas(labels: np.ndarray) -> dict[int, int]:
    """Get area of each labeled region.

    Args:
        labels: Label array of shape (H, W).

    Returns:
        Dictionary mapping label ID to area in pixels.
    """
    areas = np.bincount(labels.ravel())

    return {
        label: int(areas[label])
        for label in range(1, len(areas))
        if areas[label] > 0
    }


def get_area_statistics(labels: np.ndarray) -> dict:
    """Get statistics about region areas.

    Args:
        labels: Label array of shape (H, W).

    Returns:
        Dictionary with area statistics.
    """
    areas = np.bincount(labels.ravel())
    areas = areas[1:]  # Exclude background
    areas = areas[areas > 0]  # Only non-empty regions

    if len(areas) == 0:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0,
            "median": 0,
            "total": 0,
        }

    return {
        "count": len(areas),
        "min": int(areas.min()),
        "max": int(areas.max()),
        "mean": float(areas.mean()),
        "median": float(np.median(areas)),
        "total": int(areas.sum()),
    }
