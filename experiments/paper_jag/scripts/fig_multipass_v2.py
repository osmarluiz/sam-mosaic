"""Generate per-pass progression snapshots using the REAL multipass code.

Monkey-patches run_multipass_segmentation to capture per-pass snapshots
without modifying the source code.
"""
import sys
from pathlib import Path
import numpy as np
from scipy import ndimage
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from sam_mosaic.sam import SAMPredictor, apply_black_mask
from sam_mosaic.config import SegmentationConfig, ThresholdConfig
from sam_mosaic.points import make_uniform_grid, make_kmeans_points, make_dense_grid_points

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Config ---
checkpoint = str(Path(__file__).resolve().parents[3] / "checkpoints" / "sam2.1_hiera_large.pt")
out_base = Path(__file__).resolve().parents[1] / "output" / "multipass_progression"
out_base.mkdir(parents=True, exist_ok=True)

TILE_SIZE = 1000
PADDING = 50

data_dir = Path(__file__).resolve().parents[1] / "data"
TILES = [
    ("potsdam1_3_3",      data_dir / "potsdam/top_potsdam_3_13_RGB.tif", 3, 3, "kmeans"),
]


def load_tile_manual(img_path, row, col, tile_size, padding):
    """Load a tile with padding from a rasterio image."""
    with rasterio.open(img_path) as src:
        H, W = src.height, src.width
        nbands = min(src.count, 3)
        y0, x0 = row * tile_size, col * tile_size
        y1, x1 = min(y0 + tile_size, H), min(x0 + tile_size, W)
        py0, px0 = max(0, y0 - padding), max(0, x0 - padding)
        py1, px1 = min(H, y1 + padding), min(W, x1 + padding)
        window = rasterio.windows.Window(px0, py0, px1 - px0, py1 - py0)
        data = src.read(list(range(1, nbands + 1)), window=window).transpose(1, 2, 0)
        crop_y, crop_x = y0 - py0, x0 - px0
        useful_h, useful_w = y1 - y0, x1 - x0
    if data.dtype != np.uint8:
        if data.max() > 0:
            data = (data.astype(np.float32) / data.max() * 255).astype(np.uint8)
        else:
            data = data.astype(np.uint8)
    if data.shape[2] == 1:
        data = np.stack([data[:, :, 0]] * 3, axis=-1)
    return data, crop_x, crop_y, useful_w, useful_h


def save_snapshot(out_dir, pass_idx, orig_rgb, tile_labels, coverage, current_iou, masks_added, n_seg):
    """Save a single pass snapshot."""
    np.random.seed(42)
    n_labels = max(int(tile_labels.max()) + 1, 2)
    colors = np.random.rand(n_labels, 3)
    colors[0] = [0, 0, 0]
    colored = colors[tile_labels.astype(int) % n_labels]
    mask = tile_labels > 0
    blended = orig_rgb.astype(float) / 255.0 * 0.35 + colored * 0.65
    blended[~mask] = orig_rgb[~mask].astype(float) / 255.0
    blended = np.clip(blended, 0, 1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(blended)
    ax.set_title(
        f"Pass {pass_idx} | +{masks_added} masks | "
        f"{coverage:.1f}% coverage | {n_seg} segments | "
        f"t = {current_iou:.2f}",
        fontsize=11, fontweight='bold'
    )
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(str(out_dir / f"pass_{pass_idx:02d}.png"), dpi=200, bbox_inches='tight')
    plt.close()


def run_multipass_with_snapshots(
    predictor, image, seg_config, threshold_config,
    out_dir, orig_rgb, crop_x, crop_y, useful_w, useful_h,
    start_label=1, min_region_area=100
):
    """Exact copy of run_multipass_segmentation with snapshot hooks."""
    height, width = image.shape[:2]
    total_pixels = height * width

    combined_mask = np.zeros((height, width), dtype=np.uint8)
    tile_labels = np.zeros((height, width), dtype=np.uint32)
    current_label = start_label

    current_iou = threshold_config.iou_start
    current_stab = threshold_config.stability_start

    base_points = make_uniform_grid(height, width, seg_config.points_per_side)

    pass_idx = 0
    prev_coverage = 0.0
    working_image = None
    structure = ndimage.generate_binary_structure(2, 1)

    # Save original
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(orig_rgb)
    ax.set_title("Original tile (before segmentation)", fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(str(out_dir / "pass_00_original.png"), dpi=200, bbox_inches='tight')
    plt.close()

    while True:
        if seg_config.max_passes is not None and pass_idx >= seg_config.max_passes:
            break

        # Get points - EXACT same logic as multipass.py
        if pass_idx == 0:
            points = base_points
        else:
            if seg_config.point_strategy == "dense_grid":
                points = make_dense_grid_points(
                    combined_mask,
                    points_per_side=seg_config.points_per_side,
                    erosion_iterations=seg_config.erosion_iterations
                )
            else:
                valid_mask = combined_mask == 0
                points = make_kmeans_points(
                    valid_mask,
                    n_points=64,
                    erosion_iterations=seg_config.erosion_iterations
                )

        if len(points) == 0:
            break

        # Apply black mask - EXACT same logic
        if seg_config.use_black_mask and pass_idx > 0:
            if working_image is None:
                working_image = image.copy()
            else:
                np.copyto(working_image, image)
            current_image = apply_black_mask(working_image, combined_mask, copy=False)
        else:
            current_image = image

        # Predict
        masks = predictor.predict_points_batched(
            current_image,
            np.array(points),
            iou_threshold=current_iou,
            stability_threshold=current_stab,
            crop_n_layers=seg_config.crop_n_layers,
            box_nms_thresh=seg_config.box_nms_thresh
        )

        # Sort by area if configured
        if seg_config.sort_by_area and len(masks) > 1:
            masks.sort(key=lambda m: (m.data if hasattr(m, 'data') else m.mask).sum())

        # Add masks - EXACT same logic
        masks_added = 0
        for m in masks:
            mask_data = m.data if hasattr(m, 'data') else m.mask
            if mask_data.sum() == 0:
                continue
            labeled_mask, n_components = ndimage.label(mask_data > 0, structure=structure)
            if n_components == 0:
                continue
            if n_components == 1:
                comp_area = mask_data.sum()
                if comp_area < min_region_area:
                    continue
                overlap = (mask_data.astype(bool) & (combined_mask > 0)).sum()
                if overlap / comp_area < 0.5:
                    combined_mask[mask_data > 0] = 1
                    tile_labels[(mask_data > 0) & (tile_labels == 0)] = current_label
                    current_label += 1
                    masks_added += 1
            else:
                comp_areas = np.bincount(labeled_mask.ravel(), minlength=n_components + 1)
                for comp_id in range(1, n_components + 1):
                    comp_area = comp_areas[comp_id]
                    if comp_area < min_region_area:
                        continue
                    comp_mask = labeled_mask == comp_id
                    overlap = (comp_mask & (combined_mask > 0)).sum()
                    if overlap / comp_area < 0.5:
                        combined_mask[comp_mask] = 1
                        tile_labels[comp_mask & (tile_labels == 0)] = current_label
                        current_label += 1
                        masks_added += 1

        coverage = combined_mask.sum() / total_pixels * 100
        coverage_gain = coverage - prev_coverage

        # --- SNAPSHOT ---
        useful_labels = tile_labels[crop_y:crop_y + useful_h, crop_x:crop_x + useful_w]
        useful_mask = combined_mask[crop_y:crop_y + useful_h, crop_x:crop_x + useful_w]
        useful_cov = useful_mask.sum() / useful_mask.size * 100
        n_seg = len(np.unique(useful_labels)) - (1 if 0 in useful_labels else 0)

        print(f"  Pass {pass_idx:2d} | +{masks_added:3d} masks | {useful_cov:5.1f}% | "
              f"{n_seg} seg | t={current_iou:.2f}")

        save_snapshot(out_dir, pass_idx + 1, orig_rgb, useful_labels,
                      useful_cov, current_iou, masks_added, n_seg)

        # Stop conditions - EXACT same logic
        if coverage >= seg_config.target_coverage:
            del masks
            break

        if len(masks) == 0 or masks_added == 0:
            if current_iou <= threshold_config.iou_end:
                del masks
                break
            current_iou = max(threshold_config.iou_end, current_iou - threshold_config.step)
            current_stab = max(threshold_config.stability_end, current_stab - threshold_config.step)
        elif coverage_gain < 0.1:
            if current_iou <= threshold_config.iou_end:
                del masks
                break
            current_iou = max(threshold_config.iou_end, current_iou - threshold_config.step)
            current_stab = max(threshold_config.stability_end, current_stab - threshold_config.step)

        del masks
        prev_coverage = coverage
        pass_idx += 1

    final_cov = combined_mask.sum() / total_pixels * 100
    final_seg = current_label - start_label
    return tile_labels, combined_mask, final_cov, final_seg, pass_idx + 1


# --- Main ---
print("Loading SAM2 model...")
predictor = SAMPredictor(checkpoint_path=checkpoint)
predictor.load_model()
print("Model loaded!\n")

for tile_name, img_path, row, col, strategy in TILES:
    print(f"\n{'='*60}")
    print(f"  {tile_name} - [{row},{col}] from {img_path.name}")
    print(f"  Strategy: {strategy}")
    print(f"{'='*60}")

    out_dir = out_base / tile_name
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        image, crop_x, crop_y, useful_w, useful_h = load_tile_manual(
            img_path, row, col, TILE_SIZE, PADDING
        )
        orig_rgb = image[crop_y:crop_y + useful_h, crop_x:crop_x + useful_w]

        seg_config = SegmentationConfig(
            points_per_side=64,
            target_coverage=99.0,
            max_passes=None,
            use_black_mask=True,
            use_adaptive_threshold=True,
            point_strategy=strategy,
            erosion_iterations=0,
            crop_n_layers=0,
            box_nms_thresh=0.7,
            sort_by_area=False,
        )
        threshold_config = ThresholdConfig(
            iou_start=0.93,
            iou_end=0.60,
            stability_start=0.93,
            stability_end=0.60,
            step=0.01,
        )

        labels, mask, final_cov, final_seg, n_passes = run_multipass_with_snapshots(
            predictor, image, seg_config, threshold_config,
            out_dir, orig_rgb, crop_x, crop_y, useful_w, useful_h,
            start_label=1, min_region_area=100
        )

        print(f"  => DONE: {final_seg} segments, {final_cov:.1f}% coverage, {n_passes} passes")
        print(f"  => Saved to: {out_dir}")

    except Exception as e:
        print(f"  ERROR on {tile_name}: {e}")
        import traceback
        traceback.print_exc()

    predictor.reset_image()

print(f"\nAll done! Results in: {out_base}")
