"""Optimized merge at tile discontinuities using LUT + Union-Find.

This module provides O(n) merge complexity instead of O(n²) by:
1. Collecting all touching pairs in vectorized operations
2. Using Union-Find with path compression for transitive merges
3. Applying the merge with a single LUT lookup
"""

import numpy as np


def merge_at_discontinuities(
    labels: np.ndarray,
    tile_size: int,
    min_contact: int = 5,
    return_stats: bool = False
):
    """Merge labels at tile boundaries using optimized LUT approach.

    When tiles are processed independently, the same object may get
    different labels in adjacent tiles. This function merges labels
    that touch at tile boundaries with sufficient contact.

    The min_contact filter is applied PER discontinuity line (matching
    the original exp_plant23 behavior).

    Args:
        labels: Label array of shape (H, W).
        tile_size: Size of tiles (discontinuities at multiples of this).
        min_contact: Minimum pixels of contact PER LINE to merge (default 5).
        return_stats: If True, return (labels, stats) tuple.

    Returns:
        Label array with merged labels.
        If return_stats=True, returns (labels, stats) where stats contains:
        - pairs_found: Total touching pairs at boundaries
        - pairs_after_contact_filter: Pairs passing min_contact filter
        - unique_pairs_to_merge: Unique pairs after deduplication
        - labels_merged: Number of labels that were merged into others
    """
    height, width = labels.shape
    max_label = labels.max()

    # Initialize stats
    stats = {
        "pairs_found": 0,
        "pairs_after_contact_filter": 0,
        "unique_pairs_to_merge": 0,
        "labels_merged": 0,
    }

    if max_label == 0:
        if return_stats:
            return labels.copy(), stats
        return labels.copy()

    # Collect all touching pairs from all discontinuities
    all_pairs = []
    total_pairs_found = 0
    total_pairs_after_filter = 0

    # Vertical discontinuities (x = tile_size, 2*tile_size, ...)
    for disc_x in range(tile_size, width, tile_size):
        left_labels = labels[:, disc_x - 1]
        right_labels = labels[:, disc_x]

        # Find where different labels touch
        valid = (
            (left_labels != right_labels) &
            (left_labels > 0) &
            (right_labels > 0)
        )

        if valid.any():
            left_valid = left_labels[valid]
            right_valid = right_labels[valid]

            # Create pairs with smaller label first
            pairs_raw = np.stack([
                np.minimum(left_valid, right_valid),
                np.maximum(left_valid, right_valid)
            ], axis=1)

            # Count contacts per pair FOR THIS LINE
            unique_pairs, counts = np.unique(pairs_raw, axis=0, return_counts=True)
            total_pairs_found += len(unique_pairs)

            # Filter by min_contact PER LINE
            good_pairs = unique_pairs[counts >= min_contact]
            total_pairs_after_filter += len(good_pairs)
            if len(good_pairs) > 0:
                all_pairs.append(good_pairs)

    # Horizontal discontinuities (y = tile_size, 2*tile_size, ...)
    for disc_y in range(tile_size, height, tile_size):
        top_labels = labels[disc_y - 1, :]
        bottom_labels = labels[disc_y, :]

        valid = (
            (top_labels != bottom_labels) &
            (top_labels > 0) &
            (bottom_labels > 0)
        )

        if valid.any():
            top_valid = top_labels[valid]
            bottom_valid = bottom_labels[valid]

            pairs_raw = np.stack([
                np.minimum(top_valid, bottom_valid),
                np.maximum(top_valid, bottom_valid)
            ], axis=1)

            # Count contacts per pair FOR THIS LINE
            unique_pairs, counts = np.unique(pairs_raw, axis=0, return_counts=True)
            total_pairs_found += len(unique_pairs)

            # Filter by min_contact PER LINE
            good_pairs = unique_pairs[counts >= min_contact]
            total_pairs_after_filter += len(good_pairs)
            if len(good_pairs) > 0:
                all_pairs.append(good_pairs)

    # Update stats
    stats["pairs_found"] = total_pairs_found
    stats["pairs_after_contact_filter"] = total_pairs_after_filter

    if not all_pairs:
        if return_stats:
            return labels.copy(), stats
        return labels.copy()

    # Combine all pairs and deduplicate
    all_pairs = np.concatenate(all_pairs)
    unique_pairs = np.unique(all_pairs, axis=0)
    stats["unique_pairs_to_merge"] = len(unique_pairs)

    if len(unique_pairs) == 0:
        if return_stats:
            return labels.copy(), stats
        return labels.copy()

    # Build LUT with Union-Find
    lut = np.arange(max_label + 1, dtype=labels.dtype)

    def find_root(x: int) -> int:
        """Find root with path compression."""
        root = x
        while lut[root] != root:
            root = lut[root]
        # Path compression
        while lut[x] != root:
            next_x = lut[x]
            lut[x] = root
            x = next_x
        return root

    # Union all pairs
    for i in range(len(unique_pairs)):
        a, b = unique_pairs[i]
        root_a = find_root(a)
        root_b = find_root(b)
        if root_a != root_b:
            # Union: smaller root becomes parent
            if root_a < root_b:
                lut[root_b] = root_a
            else:
                lut[root_a] = root_b

    # Flatten LUT (ensure all point to final root)
    for i in range(len(lut)):
        lut[i] = find_root(i)

    # Count how many labels were merged (labels that point to a different root)
    labels_merged = int((lut != np.arange(len(lut))).sum())
    stats["labels_merged"] = labels_merged

    # Apply LUT in single operation
    result = lut[labels]

    if return_stats:
        return result, stats
    return result
