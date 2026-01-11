"""Merge small regions enclosed within larger regions."""

import numpy as np
from scipy import ndimage


def merge_enclosed_regions(
    labels: np.ndarray,
    max_area: int = 500
) -> np.ndarray:
    """Merge small fragments that are completely enclosed by larger masks.

    Uses connected component analysis to identify individual fragments,
    so disconnected fragments of the same label are evaluated separately.
    This matches the original exp_plant23 behavior.

    A fragment is considered enclosed if all its boundary pixels touch only
    one other label (not background).

    Fully vectorized implementation for performance.

    Args:
        labels: Label array (H, W) where each pixel has an instance ID.
        max_area: Maximum area for a fragment to be considered for merging.

    Returns:
        Label array with enclosed small fragments merged into their parent.
    """
    if labels.max() == 0:
        return labels.copy()

    # Step 1: Find connected components (each fragment gets unique ID)
    structure = ndimage.generate_binary_structure(2, 1)  # 4-connectivity
    components, n_components = ndimage.label(labels > 0, structure=structure)

    if n_components == 0:
        return labels.copy()

    # Step 2: Get area per component
    component_sizes = np.bincount(components.ravel(), minlength=n_components + 1)

    # Step 3: Find small components (vectorized)
    is_small = (component_sizes > 0) & (component_sizes <= max_area)
    is_small[0] = False  # Exclude background
    small_components = np.where(is_small)[0]

    if len(small_components) == 0:
        return labels.copy()

    # Step 4: Find ALL edge pairs (component, neighbor_label) - VECTORIZED
    # Horizontal edges
    h_diff = components[:, :-1] != components[:, 1:]
    h_left_comp = components[:, :-1][h_diff]
    h_right_comp = components[:, 1:][h_diff]
    h_left_label = labels[:, :-1][h_diff]
    h_right_label = labels[:, 1:][h_diff]

    # Vertical edges
    v_diff = components[:-1, :] != components[1:, :]
    v_top_comp = components[:-1, :][v_diff]
    v_bottom_comp = components[1:, :][v_diff]
    v_top_label = labels[:-1, :][v_diff]
    v_bottom_label = labels[1:, :][v_diff]

    # Combine all edges: (component, neighbor_label) pairs
    all_comps = np.concatenate([h_left_comp, h_right_comp, v_top_comp, v_bottom_comp])
    all_neighbor_labels = np.concatenate([h_right_label, h_left_label, v_bottom_label, v_top_label])

    # Filter to only small components (vectorized)
    is_small_mask = is_small[all_comps]
    small_comps = all_comps[is_small_mask]
    small_neighbors = all_neighbor_labels[is_small_mask]

    if len(small_comps) == 0:
        return labels.copy()

    # Step 5: For each small component, count unique neighbors (vectorized)
    pairs = np.stack([small_comps, small_neighbors], axis=1)
    unique_pairs = np.unique(pairs, axis=0)

    comp_col = unique_pairs[:, 0]
    neighbor_col = unique_pairs[:, 1]

    # Get original label for each component
    comp_to_label = np.zeros(n_components + 1, dtype=labels.dtype)
    comp_to_label[components.ravel()] = labels.ravel()

    # Check if touches background (neighbor_col == 0)
    touches_bg_pairs = unique_pairs[neighbor_col == 0]
    touches_bg = np.zeros(n_components + 1, dtype=bool)
    if len(touches_bg_pairs) > 0:
        touches_bg[touches_bg_pairs[:, 0]] = True

    # Filter pairs to non-background neighbors
    non_bg_pairs = unique_pairs[neighbor_col > 0]

    if len(non_bg_pairs) == 0:
        return labels.copy()

    # Count non-background neighbors per component
    non_bg_comp = non_bg_pairs[:, 0]
    non_bg_neighbor_counts = np.bincount(non_bg_comp, minlength=n_components + 1)

    # Enclosed = small AND exactly 1 non-bg neighbor AND not touching background
    is_enclosed = is_small & (non_bg_neighbor_counts == 1) & ~touches_bg

    enclosed_comps = np.where(is_enclosed)[0]

    if len(enclosed_comps) == 0:
        return labels.copy()

    # Step 6: Find the single neighbor for each enclosed component (vectorized)
    enclosed_mask = is_enclosed[non_bg_comp]
    enclosed_pairs = non_bg_pairs[enclosed_mask]

    # Build LUT: component -> target_label
    comp_lut = comp_to_label.copy()
    comp_lut[enclosed_pairs[:, 0]] = enclosed_pairs[:, 1]

    # Step 7: Apply using component-based LUT (single pass)
    result = comp_lut[components]
    result[components == 0] = 0

    return result


def merge_small_enclosed_iteratively(
    labels: np.ndarray,
    max_area: int = 500,
    max_iterations: int = 10
) -> np.ndarray:
    """Repeatedly merge enclosed regions until no more changes.

    Some enclosed regions may only become exposed after their
    enclosing region is merged. This iterates until convergence.

    Args:
        labels: Label array of shape (H, W).
        max_area: Maximum area for enclosed region merging.
        max_iterations: Maximum iterations to prevent infinite loops.

    Returns:
        Label array with all enclosed regions merged.
    """
    result = labels.copy()

    for _ in range(max_iterations):
        n_before = len(np.unique(result)) - 1
        result = merge_enclosed_regions(result, max_area)
        n_after = len(np.unique(result)) - 1

        if n_after == n_before:
            break

    return result
