"""Generate per-pass progression snapshots for 5 selected tiles.

For each tile, saves a PNG after every pass showing:
- Original image (dim) with colored segments overlaid
- Coverage %, pass number, threshold, masks added

Output: experiments/paper_jag/output/multipass_progression/<tile_name>/
"""
import sys
from pathlib import Path
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from sam_mosaic.sam import SAMPredictor, apply_black_mask
from sam_mosaic.points import make_uniform_grid, make_dense_grid_points, make_kmeans_points
import rasterio

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- Config ---
checkpoint = str(Path(__file__).resolve().parents[3] / "checkpoints" / "sam2.1_hiera_large.pt")
out_base = Path(__file__).resolve().parents[1] / "output" / "multipass_progression"
out_base.mkdir(parents=True, exist_ok=True)

TILE_SIZE = 1000
PADDING = 50
POINTS_PER_SIDE = 64
IOU_START, IOU_END = 0.93, 0.60
STAB_START, STAB_END = 0.93, 0.60
STEP = 0.01
TARGET_COVERAGE = 99.0
MIN_REGION_AREA = 100
EROSION_ITERATIONS = 0
BOX_NMS_THRESH = 0.7
CROP_N_LAYERS = 0

structure = ndimage.generate_binary_structure(2, 1)

# 5 tiles: (name, image_path, row, col, strategy, tile_size)
# BSB-1: 8000x8000 at T=1000 -> 8x8 grid (rows 0-7, cols 0-7)
# Potsdam-1: 6000x6000 at T=1000 -> 6x6 grid (rows 0-5, cols 0-5)
# Plant23: 10000x10000 at T=1000 -> 10x10 grid (rows 0-9, cols 0-9)
data_dir = Path(__file__).resolve().parents[1] / "data"
TILES = [
    # ("bsb1_residential",  data_dir / "brasilia/bsb_1.tif",              3, 4, "dense_grid", 1000),  # DONE
    # ("bsb1_mixed",        data_dir / "brasilia/bsb_1.tif",              5, 2, "dense_grid", 1000),  # DONE
    ("potsdam1_urban",    data_dir / "potsdam/top_potsdam_3_13_RGB.tif", 2, 3, "kmeans",     1000),
    ("plant23_fields",    data_dir / "plant23/plant23_10k.tif",          4, 5, "kmeans",     1000),
    ("plant23_lake",      data_dir / "plant23/plant23_10k.tif",          7, 3, "kmeans",     1000),
]


def load_tile_manual(img_path, row, col, tile_size, padding):
    """Load a tile with padding from a rasterio image."""
    with rasterio.open(img_path) as src:
        H, W = src.height, src.width
        nbands = min(src.count, 3)

        # Core tile coords
        y0 = row * tile_size
        x0 = col * tile_size
        y1 = min(y0 + tile_size, H)
        x1 = min(x0 + tile_size, W)

        # Padded window
        py0 = max(0, y0 - padding)
        px0 = max(0, x0 - padding)
        py1 = min(H, y1 + padding)
        px1 = min(W, x1 + padding)

        window = rasterio.windows.Window(px0, py0, px1 - px0, py1 - py0)
        data = src.read(list(range(1, nbands + 1)), window=window).transpose(1, 2, 0)

        # Crop offsets (where the useful area starts in the padded image)
        crop_y = y0 - py0
        crop_x = x0 - px0
        useful_h = y1 - y0
        useful_w = x1 - x0

    if data.dtype != np.uint8:
        if data.max() > 0:
            data = (data.astype(np.float32) / data.max() * 255).astype(np.uint8)
        else:
            data = data.astype(np.uint8)

    if data.shape[2] == 1:
        data = np.stack([data[:,:,0]]*3, axis=-1)

    return data, crop_x, crop_y, useful_w, useful_h


def save_pass_snapshot(out_dir, pass_idx, orig_rgb, tile_labels, combined_mask,
                       masks_added, coverage, current_iou, n_total_seg):
    """Save a single pass snapshot."""
    h, w = orig_rgb.shape[:2]

    np.random.seed(42)
    n_labels = int(tile_labels.max()) + 1
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
        f"{coverage:.1f}% coverage | {n_total_seg} segments | "
        f"t = {current_iou:.2f}",
        fontsize=11, fontweight='bold'
    )
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(str(out_dir / f"pass_{pass_idx:02d}.png"), dpi=200, bbox_inches='tight')
    plt.close()


def save_original(out_dir, orig_rgb):
    """Save original tile image."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(orig_rgb)
    ax.set_title("Original tile (before segmentation)", fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(str(out_dir / "pass_00_original.png"), dpi=200, bbox_inches='tight')
    plt.close()


def process_tile_with_snapshots(predictor, tile_name, img_path, row, col, strategy, tile_size):
    """Run multi-pass on a single tile, saving snapshots after each pass."""
    print(f"\n{'='*60}")
    print(f"  {tile_name} - [{row},{col}] from {img_path.name}")
    print(f"  Strategy: {strategy}, tile_size: {tile_size}")
    print(f"{'='*60}")

    out_dir = out_base / tile_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load tile
    image, crop_x, crop_y, useful_w, useful_h = load_tile_manual(
        img_path, row, col, tile_size, PADDING
    )
    h, w = image.shape[:2]
    total_pixels = h * w

    # Save original (useful area only)
    useful_rgb = image[crop_y:crop_y+useful_h, crop_x:crop_x+useful_w]
    save_original(out_dir, useful_rgb)

    # Per-tile state
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    tile_labels = np.zeros((h, w), dtype=np.uint32)
    current_label = 1
    current_iou = IOU_START
    current_stab = STAB_START
    base_points = make_uniform_grid(h, w, POINTS_PER_SIDE)
    working_image = None
    prev_coverage = 0.0

    pass_idx = 0
    while True:
        # Points
        if pass_idx == 0:
            points = base_points
        else:
            if strategy == "dense_grid":
                points = make_dense_grid_points(
                    combined_mask,
                    points_per_side=POINTS_PER_SIDE,
                    erosion_iterations=EROSION_ITERATIONS
                )
            else:
                points = make_kmeans_points(
                    combined_mask == 0,  # valid_mask: True = unsegmented
                    n_points=POINTS_PER_SIDE,
                )

        if len(points) == 0:
            print(f"  Pass {pass_idx}: no points, stopping")
            break

        # Black mask
        if pass_idx > 0:
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
            crop_n_layers=CROP_N_LAYERS,
            box_nms_thresh=BOX_NMS_THRESH
        )

        # Apply masks
        masks_added = 0
        for m in masks:
            mask_data = m.data if hasattr(m, 'data') else m.mask
            if mask_data.sum() == 0:
                continue

            labeled_mask, n_components = ndimage.label(mask_data > 0, structure=structure)
            if n_components == 0:
                continue

            comp_areas = np.bincount(labeled_mask.ravel(), minlength=n_components + 1)
            for comp_id in range(1, n_components + 1):
                comp_area = comp_areas[comp_id]
                if comp_area < MIN_REGION_AREA:
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

        # Compute useful-area stats for snapshot
        useful_labels = tile_labels[crop_y:crop_y+useful_h, crop_x:crop_x+useful_w]
        useful_mask = combined_mask[crop_y:crop_y+useful_h, crop_x:crop_x+useful_w]
        useful_cov = useful_mask.sum() / useful_mask.size * 100
        n_seg = len(np.unique(useful_labels)) - (1 if 0 in useful_labels else 0)

        print(f"  Pass {pass_idx:2d} | +{masks_added:3d} masks | {useful_cov:5.1f}% | "
              f"{n_seg} seg | t={current_iou:.2f}")

        # Save snapshot (useful area only)
        save_pass_snapshot(
            out_dir, pass_idx + 1, useful_rgb,
            useful_labels, useful_mask,
            masks_added, useful_cov, current_iou, n_seg
        )

        if useful_cov >= TARGET_COVERAGE:
            del masks
            break

        # Adaptive threshold
        if len(masks) == 0 or masks_added == 0 or coverage_gain < 0.1:
            if current_iou <= IOU_END:
                del masks
                break
            current_iou = max(IOU_END, current_iou - STEP)
            current_stab = max(STAB_END, current_stab - STEP)

        del masks
        prev_coverage = coverage
        pass_idx += 1

    predictor.reset_image()

    useful_mask_final = combined_mask[crop_y:crop_y+useful_h, crop_x:crop_x+useful_w]
    useful_labels_final = tile_labels[crop_y:crop_y+useful_h, crop_x:crop_x+useful_w]
    final_cov = useful_mask_final.sum() / useful_mask_final.size * 100
    final_seg = len(np.unique(useful_labels_final)) - (1 if 0 in useful_labels_final else 0)
    print(f"  => DONE: {final_seg} segments, {final_cov:.1f}% coverage, {pass_idx+1} passes")
    print(f"  => Saved to: {out_dir}")


# --- Main ---
print("Loading SAM2 model...")
predictor = SAMPredictor(checkpoint_path=checkpoint)
predictor.load_model()
print("Model loaded!\n")

for tile_name, img_path, row, col, strategy, ts in TILES:
    try:
        process_tile_with_snapshots(predictor, tile_name, img_path, row, col, strategy, ts)
    except Exception as e:
        print(f"  ERROR on {tile_name}: {e}")
        import traceback
        traceback.print_exc()

print(f"\nAll done! Results in: {out_base}")
