"""Optimized merge at tile discontinuities using LUT + Union-Find.

This module provides O(n) merge complexity instead of O(n²) by:
1. Collecting all touching pairs in vectorized operations
2. Using Union-Find with path compression for transitive merges
3. Applying the merge with a single LUT lookup

Merge strategies:
- "mutual_best": Parameter-free. For each boundary, only merges (A, B)
  when A is B's best match AND B is A's best match (by contact area).
- "min_contact": Merges any pair with >= min_contact pixels of contact.
"""

import numpy as np


def _build_best_match_dicts(side1_labels, side2_labels):
    """Build contact counts and best-match dicts for a boundary.

    Returns:
        (best_for_s1, best_for_s2, n_pairs_found) or (None, None, 0) if no contacts.
        best_for_s1: {s1_label: (best_s2_label, contact_count)}
        best_for_s2: {s2_label: (best_s1_label, contact_count)}
    """
    valid = (
        (side1_labels != side2_labels) &
        (side1_labels > 0) &
        (side2_labels > 0)
    )

    if not valid.any():
        return None, None, 0

    s1 = side1_labels[valid]
    s2 = side2_labels[valid]

    pairs_raw = np.stack([s1, s2], axis=1)
    unique_pairs, counts = np.unique(pairs_raw, axis=0, return_counts=True)

    best_for_s1 = {}
    best_for_s2 = {}

    for (a, b), cnt in zip(unique_pairs, counts):
        if a not in best_for_s1 or cnt > best_for_s1[a][1]:
            best_for_s1[a] = (b, int(cnt))
        if b not in best_for_s2 or cnt > best_for_s2[b][1]:
            best_for_s2[b] = (a, int(cnt))

    return best_for_s1, best_for_s2, len(unique_pairs)


def _collect_pairs_mutual_best(side1_labels, side2_labels):
    """Collect merge pairs using strict mutual best match.

    Merge (A, B) only if A's best match is B AND B's best match is A.
    Parameter-free but can miss 2:1 merges (e.g. split roof on one side).
    """
    best_s1, best_s2, n_found = _build_best_match_dicts(side1_labels, side2_labels)
    if best_s1 is None:
        return [], 0

    good_pairs = []
    for a, (b, _) in best_s1.items():
        if b in best_s2 and best_s2[b][0] == a:
            good_pairs.append((min(a, b), max(a, b)))

    return good_pairs, n_found


def _collect_pairs_best_match(side1_labels, side2_labels):
    """Collect merge pairs using one-sided best match.

    For each segment on either side, merge with its best match on the
    other side. Handles N:1 merges (e.g. a roof split into 2 on one
    side but whole on the other). Still avoids spurious merges because
    each segment only picks ONE partner — its highest-contact neighbor.
    """
    best_s1, best_s2, n_found = _build_best_match_dicts(side1_labels, side2_labels)
    if best_s1 is None:
        return [], 0

    good_pairs = set()
    # Each s1 segment merges with its best s2 match
    for a, (b, _) in best_s1.items():
        good_pairs.add((min(a, b), max(a, b)))
    # Each s2 segment merges with its best s1 match
    for b, (a, _) in best_s2.items():
        good_pairs.add((min(a, b), max(a, b)))

    return list(good_pairs), n_found


def _collect_pairs_min_contact(side1_labels, side2_labels, min_contact):
    """Collect merge pairs using minimum contact threshold.

    Args:
        side1_labels: 1D array of labels on one side of boundary.
        side2_labels: 1D array of labels on the other side.
        min_contact: Minimum pixels of contact to merge.

    Returns:
        List of (a, b) pairs to merge (normalized: a < b), and count of
        total touching pairs found.
    """
    valid = (
        (side1_labels != side2_labels) &
        (side1_labels > 0) &
        (side2_labels > 0)
    )

    if not valid.any():
        return [], 0

    s1 = side1_labels[valid]
    s2 = side2_labels[valid]

    pairs_raw = np.stack([
        np.minimum(s1, s2),
        np.maximum(s1, s2)
    ], axis=1)

    unique_pairs, counts = np.unique(pairs_raw, axis=0, return_counts=True)
    n_pairs_found = len(unique_pairs)

    good = unique_pairs[counts >= min_contact]
    good_pairs = [(int(a), int(b)) for a, b in good]

    return good_pairs, n_pairs_found


def merge_at_discontinuities(
    labels: np.ndarray,
    tile_size: int,
    min_contact: int = 20,
    merge_strategy: str = "best_match",
    return_stats: bool = False
):
    """Merge labels at tile boundaries using optimized LUT approach.

    When tiles are processed independently, the same object may get
    different labels in adjacent tiles. This function merges labels
    that touch at tile boundaries.

    Args:
        labels: Label array of shape (H, W).
        tile_size: Size of tiles (discontinuities at multiples of this).
        min_contact: Minimum pixels of contact PER LINE to merge.
            Only used when merge_strategy="min_contact".
        merge_strategy: Strategy for deciding which pairs to merge.
            - "best_match": Parameter-free. Each segment merges with its
              highest-contact neighbor across the boundary. Handles N:1.
            - "mutual_best": Parameter-free. Merge (A, B) only if A is
              B's best match and B is A's best match. Strict 1:1 only.
            - "min_contact": Merge any pair with >= min_contact pixels.
        return_stats: If True, return (labels, stats) tuple.

    Returns:
        Label array with merged labels.
        If return_stats=True, returns (labels, stats) where stats contains:
        - pairs_found: Total touching pairs at boundaries
        - pairs_after_filter: Pairs passing the merge strategy filter
        - unique_pairs_to_merge: Unique pairs after deduplication
        - labels_merged: Number of labels that were merged into others
    """
    height, width = labels.shape
    max_label = labels.max()

    # Initialize stats
    stats = {
        "pairs_found": 0,
        "pairs_after_filter": 0,
        "unique_pairs_to_merge": 0,
        "labels_merged": 0,
        "merge_strategy": merge_strategy,
    }

    if max_label == 0:
        if return_stats:
            return labels.copy(), stats
        return labels.copy()

    # Collect all touching pairs from all discontinuities
    all_pairs = []
    total_pairs_found = 0
    total_pairs_after_filter = 0

    def _process_boundary(side1, side2):
        nonlocal total_pairs_found, total_pairs_after_filter
        if merge_strategy == "best_match":
            pairs, n_found = _collect_pairs_best_match(side1, side2)
        elif merge_strategy == "mutual_best":
            pairs, n_found = _collect_pairs_mutual_best(side1, side2)
        else:
            pairs, n_found = _collect_pairs_min_contact(side1, side2, min_contact)
        total_pairs_found += n_found
        total_pairs_after_filter += len(pairs)
        if pairs:
            all_pairs.extend(pairs)

    # Vertical discontinuities (x = tile_size, 2*tile_size, ...)
    for disc_x in range(tile_size, width, tile_size):
        _process_boundary(labels[:, disc_x - 1], labels[:, disc_x])

    # Horizontal discontinuities (y = tile_size, 2*tile_size, ...)
    for disc_y in range(tile_size, height, tile_size):
        _process_boundary(labels[disc_y - 1, :], labels[disc_y, :])

    # Update stats
    stats["pairs_found"] = total_pairs_found
    stats["pairs_after_filter"] = total_pairs_after_filter

    if not all_pairs:
        if return_stats:
            return labels.copy(), stats
        return labels.copy()

    # Deduplicate pairs across all boundary lines
    unique_pairs_set = set(all_pairs)
    unique_pairs = np.array(list(unique_pairs_set), dtype=np.uint32) if unique_pairs_set else np.array([], dtype=np.uint32).reshape(0, 2)
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

    # Flatten LUT only for labels that exist in the image (skip unused label IDs)
    # This is O(unique_labels) instead of O(max_label)
    used_labels = np.unique(labels)
    used_labels = used_labels[used_labels > 0]  # Exclude background
    for label in used_labels:
        lut[label] = find_root(label)

    # Count how many labels were merged (labels that point to a different root)
    labels_merged = int((lut != np.arange(len(lut))).sum())
    stats["labels_merged"] = labels_merged

    # Apply LUT in single operation
    result = lut[labels]

    if return_stats:
        return result, stats
    return result
