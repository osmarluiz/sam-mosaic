"""Cascade refinement for progressive segmentation."""

from typing import List, Callable, Optional, Tuple
from pathlib import Path
import numpy as np
from tqdm import tqdm

from sam_mosaic.segmentation.sam import SAMPredictor, Mask, apply_black_mask
from sam_mosaic.config import CascadeConfig


def run_single_pass(
    predictor: SAMPredictor,
    image: np.ndarray,
    combined_mask: np.ndarray,
    initial_points: np.ndarray,
    config: CascadeConfig,
    pass_idx: int,
    mask_filter: Optional[Callable[[Mask], bool]] = None,
) -> Tuple[List[Mask], np.ndarray, float]:
    """Run a single pass of cascade refinement.

    Args:
        predictor: SAM predictor instance.
        image: RGB image array (H, W, 3).
        combined_mask: Current combined mask (H, W) - areas already segmented.
        initial_points: Initial point grid for pass 0 (N, 2).
        config: Cascade configuration.
        pass_idx: Current pass index (0-based).
        mask_filter: Optional function to filter masks.

    Returns:
        Tuple of (new_masks, updated_combined_mask, coverage_percent).
    """
    from sam_mosaic.points.grids import make_kmeans_points

    # Get thresholds for this pass
    iou_thresh, stability_thresh = config.thresholds.interpolate(
        pass_idx, config.n_passes
    )

    # Select points based on pass number
    if pass_idx == 0:
        # Pass 0: Use initial uniform grid
        points = initial_points
    else:
        # Pass 1+: K-means in unmasked areas
        valid_mask = combined_mask == 0
        points = make_kmeans_points(
            valid_mask,
            n_points=config.points_per_pass,
            erosion=config.point_erosion
        )

    if len(points) == 0:
        coverage = combined_mask.sum() / combined_mask.size * 100
        return [], combined_mask, coverage

    # Apply black mask to image for this pass
    current_image = apply_black_mask(image, combined_mask)

    # Generate masks using batched prediction (MUCH faster)
    masks = predictor.predict_points_batched(
        current_image,
        points,
        iou_thresh=iou_thresh,
        stability_thresh=stability_thresh
    )

    # Apply optional filter
    if mask_filter is not None:
        masks = [m for m in masks if mask_filter(m)]

    # Update combined mask
    for mask in masks:
        combined_mask = np.maximum(combined_mask, mask.mask)

    coverage = combined_mask.sum() / combined_mask.size * 100

    return masks, combined_mask, coverage


def save_combined_mask(mask: np.ndarray, path: Path) -> None:
    """Save combined mask to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, mask)


def load_combined_mask(path: Path, height: int, width: int) -> np.ndarray:
    """Load combined mask from disk, or create empty if not exists."""
    if path.exists():
        return np.load(path)
    return np.zeros((height, width), dtype=np.uint8)


def run_border_cascade(
    predictor: SAMPredictor,
    image: np.ndarray,
    points: np.ndarray,
    config: CascadeConfig,
    mask_filter: Callable[[Mask], bool],
    save_passes: Optional[Path] = None,
    tile_id: str = ""
) -> List[Mask]:
    """Run cascade for border correction tiles (V, H, corners).

    Unlike run_cascade, this uses a FIXED grid of points for all passes.
    Points are NOT removed when they fall inside already-segmented areas.
    The image is blacked out in segmented regions, and the crossing filter
    ensures only masks that cross the discontinuity are accepted.

    Args:
        predictor: SAM predictor instance.
        image: RGB image array (H, W, 3).
        points: Fixed point grid used for ALL passes (N, 2).
        config: Cascade configuration.
        mask_filter: Function to filter masks (crosses_v, crosses_h, crosses_corner).
        save_passes: Optional path to save intermediate pass results.
        tile_id: Tile identifier for saving.

    Returns:
        List of all accepted masks across all passes.
    """
    all_masks: List[Mask] = []
    combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    pbar = tqdm(range(config.n_passes), desc=f"  Passes", leave=False, unit="pass")

    for pass_idx in pbar:
        # Get thresholds for this pass (relax progressively)
        iou_thresh, stability_thresh = config.thresholds.interpolate(
            pass_idx, config.n_passes
        )

        # Apply black mask to image for this pass
        current_image = apply_black_mask(image, combined_mask)

        # Always use the SAME fixed grid of points
        masks = predictor.predict_points_batched(
            current_image,
            points,
            iou_thresh=iou_thresh,
            stability_thresh=stability_thresh
        )

        # Filter only masks that cross the discontinuity
        masks = [m for m in masks if mask_filter(m)]

        # Update combined mask
        for mask in masks:
            combined_mask = np.maximum(combined_mask, mask.mask)

        all_masks.extend(masks)

        coverage = combined_mask.sum() / combined_mask.size * 100

        pbar.set_postfix({
            "masks": len(all_masks),
            "cov": f"{coverage:.1f}%",
            "iou": f"{iou_thresh:.2f}"
        })

        # Save intermediate pass result
        if save_passes is not None:
            save_passes.mkdir(parents=True, exist_ok=True)
            from sam_mosaic.io.writer import save_mask
            save_mask(
                combined_mask,
                save_passes / f"{tile_id}_pass{pass_idx:02d}.tif"
            )

        # Early stop if no new masks found
        if len(masks) == 0 and pass_idx > 0:
            pbar.set_postfix({"status": "complete"})
            break

    return all_masks


def run_cascade(
    predictor: SAMPredictor,
    image: np.ndarray,
    initial_points: np.ndarray,
    config: CascadeConfig,
    mask_filter: Optional[Callable[[Mask], bool]] = None,
    save_passes: Optional[Path] = None,
    tile_id: str = ""
) -> List[Mask]:
    """Run cascade refinement over multiple passes (legacy single-tile version).

    Pass 0: Uses the initial uniform grid.
    Pass 1+: Uses K-means to distribute points in unmasked areas.

    Each pass uses progressively relaxed thresholds. Already-segmented
    regions are masked (blacked out) before each subsequent pass.

    Args:
        predictor: SAM predictor instance.
        image: RGB image array (H, W, 3).
        initial_points: Initial point grid for pass 0 (N, 2).
        config: Cascade configuration.
        mask_filter: Optional function to filter masks (e.g., crosses_v).
        save_passes: Optional path to save intermediate pass results.
        tile_id: Tile identifier for saving.

    Returns:
        List of all accepted masks across all passes.
    """
    all_masks: List[Mask] = []
    combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    pbar = tqdm(range(config.n_passes), desc=f"  Passes", leave=False, unit="pass")

    for pass_idx in pbar:
        masks, combined_mask, coverage = run_single_pass(
            predictor=predictor,
            image=image,
            combined_mask=combined_mask,
            initial_points=initial_points,
            config=config,
            pass_idx=pass_idx,
            mask_filter=mask_filter
        )

        all_masks.extend(masks)

        # Get thresholds for progress bar
        iou_thresh, _ = config.thresholds.interpolate(pass_idx, config.n_passes)

        pbar.set_postfix({
            "masks": len(all_masks),
            "cov": f"{coverage:.1f}%",
            "iou": f"{iou_thresh:.2f}"
        })

        # Save intermediate pass result
        if save_passes is not None:
            save_passes.mkdir(parents=True, exist_ok=True)
            from sam_mosaic.io.writer import save_mask
            save_mask(
                combined_mask,
                save_passes / f"{tile_id}_pass{pass_idx:02d}.tif"
            )

        # Early stop if coverage is very high
        if pass_idx > 0 and coverage > 99.5:
            pbar.set_postfix({"status": "converged"})
            break

        # Early stop if no points left
        if len(masks) == 0 and pass_idx > 0:
            pbar.set_postfix({"status": "complete"})
            break

    return all_masks


def run_cascade_on_tile(
    predictor: SAMPredictor,
    tile_image: np.ndarray,
    points: np.ndarray,
    config: CascadeConfig,
    tile_x: int = 0,
    tile_y: int = 0,
    disc_type: str = "base",
    disc_x: int = 0,
    disc_y: int = 0,
    save_passes: Optional[Path] = None,
    tile_id: str = ""
) -> List[Mask]:
    """Run cascade on a single tile with optional border filtering.

    Args:
        predictor: SAM predictor instance.
        tile_image: RGB tile image (H, W, 3).
        points: Point grid in tile coordinates (N, 2).
        config: Cascade configuration.
        tile_x: Tile x-offset in global coordinates.
        tile_y: Tile y-offset in global coordinates.
        disc_type: Discontinuity type ("base", "v", "h", "corner").
        disc_x: X-coordinate of discontinuity line.
        disc_y: Y-coordinate of discontinuity line.
        save_passes: Optional path to save intermediate pass results.
        tile_id: Tile identifier for saving.

    Returns:
        List of masks (in tile coordinates).
    """
    # Base tiles: use K-means cascade (original behavior)
    if disc_type == "base":
        return run_cascade(
            predictor=predictor,
            image=tile_image,
            points=points,
            config=config,
            mask_filter=None,
            save_passes=save_passes,
            tile_id=tile_id
        )

    # Border tiles (v, h, corner): use fixed grid cascade
    # Create mask filter based on discontinuity type
    if disc_type == "v":
        from sam_mosaic.tiling.borders import crosses_v
        mask_filter = lambda m: crosses_v(m.mask, disc_x, tile_x)

    elif disc_type == "h":
        from sam_mosaic.tiling.borders import crosses_h
        mask_filter = lambda m: crosses_h(m.mask, disc_y, tile_y)

    else:  # corner
        from sam_mosaic.tiling.borders import crosses_corner
        mask_filter = lambda m: crosses_corner(m.mask, disc_x, disc_y, tile_x, tile_y)

    return run_border_cascade(
        predictor=predictor,
        image=tile_image,
        points=points,
        config=config,
        mask_filter=mask_filter,
        save_passes=save_passes,
        tile_id=tile_id
    )


def masks_to_labels(
    masks: List[Mask],
    height: int,
    width: int
) -> np.ndarray:
    """Convert list of masks to label image.

    Later masks overwrite earlier ones (useful for priority-based merging).

    Args:
        masks: List of Mask objects.
        height: Output height.
        width: Output width.

    Returns:
        Label array (H, W) where each pixel has an instance ID (0 = background).
    """
    labels = np.zeros((height, width), dtype=np.uint32)

    for i, mask in enumerate(masks, start=1):
        labels[mask.mask > 0] = i

    return labels


def labels_to_binary(labels: np.ndarray) -> np.ndarray:
    """Convert label image to binary mask.

    Args:
        labels: Label array (H, W).

    Returns:
        Binary mask where non-zero labels become 1.
    """
    return (labels > 0).astype(np.uint8)
