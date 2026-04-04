"""Run multipass on 4 candidate Potsdam tiles to pick the best one."""
import sys
from pathlib import Path
import numpy as np
from scipy import ndimage
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from sam_mosaic.sam import SAMPredictor, apply_black_mask
from sam_mosaic.config import SegmentationConfig, ThresholdConfig
from sam_mosaic.points import make_uniform_grid, make_kmeans_points

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

checkpoint = str(Path(__file__).resolve().parents[3] / "checkpoints" / "sam2.1_hiera_large.pt")
out_base = Path(__file__).resolve().parents[1] / "output" / "multipass_progression"
data_dir = Path(__file__).resolve().parents[1] / "data"
img_path = data_dir / "potsdam" / "top_potsdam_3_13_RGB.tif"

TILE_SIZE = 1000
PADDING = 50

candidates = [
    ("potsdam_1_1", 1, 1),
    ("potsdam_2_1", 2, 1),
    ("potsdam_3_3", 3, 3),
    ("potsdam_4_2", 4, 2),
]


def load_tile(img_path, row, col):
    with rasterio.open(img_path) as src:
        H, W = src.height, src.width
        y0, x0 = row * TILE_SIZE, col * TILE_SIZE
        y1, x1 = min(y0 + TILE_SIZE, H), min(x0 + TILE_SIZE, W)
        py0, px0 = max(0, y0 - PADDING), max(0, x0 - PADDING)
        py1, px1 = min(H, y1 + PADDING), min(W, x1 + PADDING)
        window = rasterio.windows.Window(px0, py0, px1 - px0, py1 - py0)
        data = src.read([1, 2, 3], window=window).transpose(1, 2, 0)
        crop_y, crop_x = y0 - py0, x0 - px0
        useful_h, useful_w = y1 - y0, x1 - x0
    return data, crop_x, crop_y, useful_w, useful_h


def run_and_summarize(predictor, name, row, col):
    print(f"\n=== {name} [{row},{col}] ===")
    image, crop_x, crop_y, uw, uh = load_tile(img_path, row, col)
    h, w = image.shape[:2]

    seg_config = SegmentationConfig(
        points_per_side=64, target_coverage=99.0, max_passes=None,
        use_black_mask=True, use_adaptive_threshold=True,
        point_strategy="kmeans", erosion_iterations=0,
        crop_n_layers=0, box_nms_thresh=0.7, sort_by_area=False,
    )
    threshold_config = ThresholdConfig(
        iou_start=0.93, iou_end=0.60,
        stability_start=0.93, stability_end=0.60, step=0.01,
    )

    combined_mask = np.zeros((h, w), dtype=np.uint8)
    tile_labels = np.zeros((h, w), dtype=np.uint32)
    current_label = 1
    current_iou = 0.93
    current_stab = 0.93
    base_points = make_uniform_grid(h, w, 64)
    working_image = None
    prev_coverage = 0.0
    structure = ndimage.generate_binary_structure(2, 1)

    coverages = []
    pass_idx = 0

    while True:
        if pass_idx == 0:
            points = base_points
        else:
            valid_mask = combined_mask == 0
            points = make_kmeans_points(valid_mask, n_points=64, erosion_iterations=0)

        if len(points) == 0:
            break

        if pass_idx > 0:
            if working_image is None:
                working_image = image.copy()
            else:
                np.copyto(working_image, image)
            current_image = apply_black_mask(working_image, combined_mask, copy=False)
        else:
            current_image = image

        masks = predictor.predict_points_batched(
            current_image, np.array(points),
            iou_threshold=current_iou, stability_threshold=current_stab,
            crop_n_layers=0, box_nms_thresh=0.7
        )

        masks_added = 0
        for m in masks:
            mask_data = m.data if hasattr(m, 'data') else m.mask
            if mask_data.sum() == 0:
                continue
            labeled_mask, n_comp = ndimage.label(mask_data > 0, structure=structure)
            if n_comp == 0:
                continue
            comp_areas = np.bincount(labeled_mask.ravel(), minlength=n_comp + 1)
            for cid in range(1, n_comp + 1):
                if comp_areas[cid] < 100:
                    continue
                comp_mask = labeled_mask == cid
                overlap = (comp_mask & (combined_mask > 0)).sum()
                if overlap / comp_areas[cid] < 0.5:
                    combined_mask[comp_mask] = 1
                    tile_labels[comp_mask & (tile_labels == 0)] = current_label
                    current_label += 1
                    masks_added += 1

        useful_mask = combined_mask[crop_y:crop_y+uh, crop_x:crop_x+uw]
        cov = useful_mask.sum() / useful_mask.size * 100
        coverages.append(cov)
        cov_gain = cov - prev_coverage

        if pass_idx % 5 == 0 or cov > 95:
            print(f"  Pass {pass_idx:2d} | +{masks_added:3d} | {cov:5.1f}% | t={current_iou:.2f}")

        if cov >= 99.0:
            del masks
            break

        if len(masks) == 0 or masks_added == 0 or cov_gain < 0.1:
            if current_iou <= 0.60:
                del masks
                break
            current_iou = max(0.60, current_iou - 0.01)
            current_stab = max(0.60, current_stab - 0.01)

        del masks
        prev_coverage = cov
        pass_idx += 1

    predictor.reset_image()
    print(f"  FINAL: {cov:.1f}% in {pass_idx+1} passes, {current_label-1} segments")
    return coverages


print("Loading SAM2...")
predictor = SAMPredictor(checkpoint_path=checkpoint)
predictor.load_model()
print("OK\n")

all_covs = {}
for name, row, col in candidates:
    try:
        covs = run_and_summarize(predictor, name, row, col)
        all_covs[name] = covs
    except Exception as e:
        print(f"  ERROR: {e}")

# Plot comparison
fig, ax = plt.subplots(figsize=(10, 5))
for name, covs in all_covs.items():
    ax.plot(range(len(covs)), covs, '-o', ms=3, label=name)
ax.set_xlabel('Pass')
ax.set_ylabel('Coverage (%)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(str(out_base / "potsdam_candidates.png"), dpi=150)
plt.close()
print(f"\nSaved comparison: {out_base / 'potsdam_candidates.png'}")
