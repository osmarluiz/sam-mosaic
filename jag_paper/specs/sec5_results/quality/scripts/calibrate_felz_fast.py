"""Fast Felzenszwalb calibration: binary search on crop, verify on full image.

Optimizations:
- flush=True on all prints for real-time progress
- Smaller crop (1500x1500) for faster binary search
- Use multichannel=True explicitly
- Only 2 verification runs on full image instead of 3
"""
import numpy as np
import rasterio
from skimage.segmentation import felzenszwalb
import time, sys

def p(msg):
    print(msg, flush=True)

datasets = [
    ('BSB-1', 'experiments/paper_jag/data/brasilia/bsb_1.tif', 27427),
    ('Potsdam', 'experiments/paper_jag/data/potsdam/top_potsdam_3_13_RGB.tif', 2388),
    ('Plant23', 'experiments/paper_jag/data/plant23/plant23_10k.tif', 3530),
]

CROP = 1500

for name, path, target_seg in datasets:
    p(f"\n{'='*50}")
    p(f"  {name} (target: {target_seg} segments)")
    p(f"{'='*50}")

    t_load = time.time()
    with rasterio.open(path) as src:
        H, W = src.height, src.width
        img_full = src.read([1, 2, 3]).transpose(1, 2, 0)

    if img_full.dtype != np.uint8:
        if img_full.max() > 0:
            img_full = (img_full.astype(np.float32) / img_full.max() * 255).astype(np.uint8)
    p(f"  Loaded {H}x{W} in {time.time()-t_load:.1f}s")

    # Center crop
    cy, cx = H // 2, W // 2
    c2 = CROP // 2
    crop = img_full[cy-c2:cy+c2, cx-c2:cx+c2]
    target_crop = int(target_seg * CROP * CROP / (H * W))
    p(f"  Crop target: ~{target_crop} segments")

    # Binary search on crop
    lo, hi = 10, 5000
    best_scale = 200
    best_diff = float('inf')

    t0 = time.time()
    for i in range(15):
        scale = (lo + hi) / 2
        labels = felzenszwalb(crop, scale=scale, sigma=0.5, min_size=50)
        n_seg = len(np.unique(labels))
        diff = abs(n_seg - target_crop)
        p(f"    iter {i}: scale={scale:.0f} -> {n_seg} seg (target {target_crop}, diff={diff})")

        if diff < best_diff:
            best_diff = diff
            best_scale = scale

        if n_seg > target_crop:
            lo = scale
        else:
            hi = scale

        if diff < target_crop * 0.05:
            break

    p(f"  Crop done in {time.time()-t0:.1f}s, best scale={best_scale:.0f}")

    # Single verification on full image
    p(f"  Running full image with scale={best_scale:.0f}...")
    t1 = time.time()
    labels_full = felzenszwalb(img_full, scale=best_scale, sigma=0.5, min_size=50)
    n_seg_full = len(np.unique(labels_full))
    p(f"  Full: scale={best_scale:.0f} -> {n_seg_full} seg (target {target_seg}) [{time.time()-t1:.0f}s]")

    # If off by >15%, try one adjustment
    ratio = n_seg_full / target_seg
    if ratio < 0.85 or ratio > 1.15:
        # Adjust scale proportionally
        adjusted = best_scale * (n_seg_full / target_seg) ** 0.5
        p(f"  Adjusting: scale={adjusted:.0f}...")
        t2 = time.time()
        labels_full = felzenszwalb(img_full, scale=adjusted, sigma=0.5, min_size=50)
        n_seg_full = len(np.unique(labels_full))
        p(f"  Full: scale={adjusted:.0f} -> {n_seg_full} seg [{time.time()-t2:.0f}s]")
        best_scale = adjusted

    p(f"\n  RESULT: scale={best_scale:.0f}, {n_seg_full} segments (target {target_seg})")

p("\nDone!")
