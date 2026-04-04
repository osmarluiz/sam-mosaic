"""Final multi-pass progression figure with per-pass coloring.

Changes from v2:
- Segments colored by pass index (same color for all segments in same pass)
- Info overlay inside images (not as title)
- Colored border around each image matching the curve color
- No dataset label on side
- Larger fonts
- Tight vertical spacing
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
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

checkpoint = str(Path(__file__).resolve().parents[3] / "checkpoints" / "sam2.1_hiera_large.pt")
out_base = Path(__file__).resolve().parents[1] / "output" / "multipass_progression"
fig_dir = Path(__file__).resolve().parents[1] / "submission" / "latex" / "figures"
data_dir = Path(__file__).resolve().parents[1] / "data"

TILE_SIZE = 1000
PADDING = 50

# Generate a colormap: one distinct color per pass
def make_pass_colors(max_passes=100):
    """Create distinct colors for each pass index."""
    cmap = plt.cm.get_cmap('tab20', 20)
    colors = []
    for i in range(max_passes):
        colors.append(cmap(i % 20)[:3])
    return colors

PASS_COLORS = make_pass_colors(100)


def load_tile(img_path, row, col):
    with rasterio.open(img_path) as src:
        H, W = src.height, src.width
        nbands = min(src.count, 3)
        y0, x0 = row * TILE_SIZE, col * TILE_SIZE
        y1, x1 = min(y0 + TILE_SIZE, H), min(x0 + TILE_SIZE, W)
        py0, px0 = max(0, y0 - PADDING), max(0, x0 - PADDING)
        py1, px1 = min(H, y1 + PADDING), min(W, x1 + PADDING)
        window = rasterio.windows.Window(px0, py0, px1 - px0, py1 - py0)
        data = src.read(list(range(1, nbands + 1)), window=window).transpose(1, 2, 0)
        crop_y, crop_x = y0 - py0, x0 - px0
        uh, uw = y1 - y0, x1 - x0
    if data.dtype != np.uint8:
        if data.max() > 0:
            data = (data.astype(np.float32) / data.max() * 255).astype(np.uint8)
        else:
            data = data.astype(np.uint8)
    if data.shape[2] == 1:
        data = np.stack([data[:,:,0]]*3, axis=-1)
    return data, crop_x, crop_y, uw, uh


def run_multipass_passcolored(predictor, image, strategy, crop_x, crop_y, uw, uh):
    """Run multipass, return pass_map (which pass each pixel was segmented in)."""
    h, w = image.shape[:2]
    total_pixels = h * w

    combined_mask = np.zeros((h, w), dtype=np.uint8)
    tile_labels = np.zeros((h, w), dtype=np.uint32)
    pass_map = np.zeros((h, w), dtype=np.int32)  # -1 = unsegmented, 0+ = pass index
    pass_map[:] = -1
    current_label = 1
    current_iou = 0.93
    current_stab = 0.93
    base_points = make_uniform_grid(h, w, 64)
    working_image = None
    prev_coverage = 0.0
    structure = ndimage.generate_binary_structure(2, 1)

    # Store per-pass snapshots of the useful area
    snapshots = []  # list of (pass_idx, coverage, threshold, n_seg, pass_map_useful)

    pass_idx = 0
    while True:
        if pass_idx == 0:
            points = base_points
        else:
            if strategy == "dense_grid":
                points = make_dense_grid_points(combined_mask, points_per_side=64, erosion_iterations=0)
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
                    pass_map[comp_mask & (pass_map == -1)] = pass_idx
                    current_label += 1
                    masks_added += 1

        useful_mask = combined_mask[crop_y:crop_y+uh, crop_x:crop_x+uw]
        useful_pass = pass_map[crop_y:crop_y+uh, crop_x:crop_x+uw]
        useful_labels = tile_labels[crop_y:crop_y+uh, crop_x:crop_x+uw]
        cov = useful_mask.sum() / useful_mask.size * 100
        n_seg = len(np.unique(useful_labels)) - (1 if 0 in useful_labels else 0)

        # Save snapshot
        snapshots.append((pass_idx, cov, current_iou, n_seg, useful_pass.copy()))

        print(f"  Pass {pass_idx:2d} | +{masks_added:3d} | {cov:5.1f}% | {n_seg} seg | t={current_iou:.2f}")

        coverage = combined_mask.sum() / total_pixels * 100
        coverage_gain = coverage - prev_coverage

        if cov >= 99.0:
            del masks
            break

        if len(masks) == 0 or masks_added == 0:
            if current_iou <= 0.60:
                del masks
                break
            current_iou = max(0.60, current_iou - 0.01)
            current_stab = max(0.60, current_stab - 0.01)
        elif coverage_gain < 0.1:
            if current_iou <= 0.60:
                del masks
                break
            current_iou = max(0.60, current_iou - 0.01)
            current_stab = max(0.60, current_stab - 0.01)

        del masks
        prev_coverage = coverage
        pass_idx += 1

    predictor.reset_image()
    return snapshots


def render_grouped(orig_rgb, pass_map, frame_breaks, current_frame_idx, alpha=0.6):
    """Render image with segments colored by frame group.

    frame_breaks: list of pass indices where each shown frame ends.
                  e.g., [29, 56, 79] means group 0 = passes 0-29,
                  group 1 = passes 30-56, group 2 = passes 57-79.
    current_frame_idx: which frame we're rendering (0, 1, 2, ...).
                       Only segments up to frame_breaks[current_frame_idx] are shown.
    """
    # 3 distinct, saturated colors for the 3 frame groups
    GROUP_COLORS = [
        (0.20, 0.60, 0.86),   # blue
        (0.90, 0.50, 0.13),   # orange
        (0.55, 0.18, 0.63),   # purple
    ]

    h, w = orig_rgb.shape[:2]
    blended = orig_rgb.astype(np.float32) / 255.0

    for gi in range(current_frame_idx + 1):
        lo = 0 if gi == 0 else frame_breaks[gi - 1] + 1
        hi = frame_breaks[gi]
        mask = (pass_map >= lo) & (pass_map <= hi)
        if mask.any():
            color = np.array(GROUP_COLORS[gi % len(GROUP_COLORS)])
            blended[mask] = blended[mask] * (1 - alpha) + color * alpha

    return np.clip(blended, 0, 1)


# === TILES ===
tiles_config = [
    ("BSB-1",       data_dir / "brasilia/bsb_1.tif",              3, 4, "dense_grid", '#c0392b'),
    ("Potsdam",     data_dir / "potsdam/top_potsdam_3_13_RGB.tif", 3, 3, "kmeans",     '#2980b9'),
    ("Agriculture", data_dir / "plant23/plant23_10k.tif",          4, 5, "kmeans",     '#27ae60'),
]

# Frame selection: Original + 3 stages (start, middle, final)
# [original, first_pass, intermediate, final]
FRAME_SELECTION = {
    "BSB-1":       [0, 0, 29, -1],        # original, pass 0 (32%), pass 29 (67%), final (88%)
    "Potsdam":     [0, 0, 15, -1],         # original, pass 0 (48%), pass 15 (64%), final (98%)
    "Agriculture": [0, 0, 8, -1],          # original, pass 0 (57%), pass 8 (84%), final (97%)
}

print("Loading SAM2...")
predictor = SAMPredictor(checkpoint_path=checkpoint)
predictor.load_model()
print("OK\n")

all_data = {}
for name, img_path, row, col, strategy, color in tiles_config:
    print(f"\n{'='*50}")
    print(f"  {name} [{row},{col}] strategy={strategy}")
    print(f"{'='*50}")

    image, crop_x, crop_y, uw, uh = load_tile(img_path, row, col)
    orig_rgb = image[crop_y:crop_y+uh, crop_x:crop_x+uw]

    snapshots = run_multipass_passcolored(predictor, image, strategy, crop_x, crop_y, uw, uh)

    coverages = [s[1] for s in snapshots]
    print(f"  DONE: {len(snapshots)} passes, final {coverages[-1]:.1f}%")

    all_data[name] = {
        'orig_rgb': orig_rgb,
        'snapshots': snapshots,
        'coverages': coverages,
        'color': color,
    }

# === BUILD FIGURE ===
print("\nBuilding figure...")

fig = plt.figure(figsize=(14, 14))
gs = GridSpec(4, 4, figure=fig, height_ratios=[1, 1, 1, 0.65],
             hspace=0.08, wspace=0.03,
             left=0.02, right=0.98, top=0.98, bottom=0.05)

border_width = 4  # pixels for colored border

for row_idx, (name, img_path, row, col, strategy, border_color) in enumerate(tiles_config):
    data = all_data[name]
    orig_rgb = data['orig_rgb']
    snapshots = data['snapshots']
    frame_indices = FRAME_SELECTION[name]

    # Resolve -1 to last
    frame_indices = [fi if fi >= 0 else len(snapshots) - 1 for fi in frame_indices]
    # Clamp
    frame_indices = [min(fi, len(snapshots) - 1) for fi in frame_indices]

    # Compute frame_breaks: the pass index at which each shown frame ends
    # frame_indices[1], [2], [3] are the snapshot list indices for the 3 segmented frames
    frame_breaks = []
    for fi in frame_indices[1:]:  # skip original (index 0)
        pidx_at_frame = snapshots[fi][0]  # pass_idx
        frame_breaks.append(pidx_at_frame)

    for col_idx in range(4):
        ax = fig.add_subplot(gs[row_idx, col_idx])

        if col_idx == 0:
            # Original image
            img = orig_rgb.astype(np.float32) / 255.0
            info_text = "Original"
        else:
            si = frame_indices[col_idx]
            pidx, cov, thresh, nseg, pass_map = snapshots[si]
            # Render with grouped colors: col_idx 1->group 0, 2->group 1, 3->group 2
            img = render_grouped(orig_rgb, pass_map, frame_breaks,
                                 current_frame_idx=col_idx - 1, alpha=0.6)
            info_text = f"Pass {pidx}  ({cov:.0f}%)"

        ax.imshow(img)
        ax.axis('off')

        # Colored border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)

        # Info text inside image (bottom-left)
        ax.text(0.03, 0.04, info_text, transform=ax.transAxes,
                fontsize=15, fontweight='bold', va='bottom', color='white',
                bbox=dict(facecolor='black', alpha=0.75,
                          boxstyle='round,pad=0.25', edgecolor='none'))

        # Panel letter (top-left) - uppercase
        letter = chr(ord('A') + row_idx * 4 + col_idx)
        ax.text(0.03, 0.97, f"{letter}", transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='top', color='white',
                bbox=dict(facecolor='black', alpha=0.8,
                          boxstyle='round,pad=0.12', edgecolor='none'))

# === Coverage curves ===
ax_c = fig.add_subplot(gs[3, :])

for name, img_path, row, col, strategy, color in tiles_config:
    data = all_data[name]
    covs = data['coverages']
    ax_c.plot(range(len(covs)), covs, '-', color=color, lw=2.5, label=name, zorder=3)

    # Markers for selected frames (skip original = index 0)
    frame_indices = FRAME_SELECTION[name]
    frame_indices = [fi if fi >= 0 else len(covs) - 1 for fi in frame_indices]
    frame_indices = [min(fi, len(covs) - 1) for fi in frame_indices]
    for fi in frame_indices[1:]:  # skip original
        ax_c.plot(fi, covs[fi], 'o', color=color, ms=10, zorder=5,
                  markeredgecolor='white', markeredgewidth=1.5)

ax_c.set_xlabel('Pass number', fontsize=15, labelpad=5)
ax_c.set_ylabel('Coverage (%)', fontsize=15, labelpad=5)
ax_c.set_xlim(-1, 82)
ax_c.set_ylim(25, 102)
ax_c.legend(fontsize=14, loc='center right', framealpha=0.95,
            edgecolor='#dddddd', fancybox=True)
ax_c.grid(True, alpha=0.15, color='#cccccc')
ax_c.tick_params(labelsize=13)
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)

out_path = fig_dir / "fig3_multipass_progression.png"
plt.savefig(str(out_path), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nSaved: {out_path}")
