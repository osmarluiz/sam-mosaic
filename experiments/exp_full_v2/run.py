#!/usr/bin/env python3
"""
Experiment: Full Plant23 image (15000 x 30000) - V2

Changes from exp_full:
  - V/H/Corners: Fixed grid (pass 0), K-means in zone (pass 1+)
  - Filter large masks (>40%) BEFORE updating combined_mask
  - Base: IoU & Stability 0.92 -> 0.78 (15 passes)
  - V/H/Corners: IoU & Stability 0.92 -> 0.88 (5 passes)
  - Zone width: 50px (was 100px)

Structure:
  exp_full_v2/
  ├── config.yaml      # Configuration
  ├── run.py           # This script
  └── output/
      ├── band0_base.tif
      ├── band1_vh.tif
      ├── band2_corners.tif
      ├── merged_labels.tif
      └── segments.shp
"""

# Silence warnings
import os
os.environ["OMP_NUM_THREADS"] = "4"

import warnings
warnings.filterwarnings("ignore", message=".*MiniBatchKMeans.*memory leak.*")
warnings.filterwarnings("ignore", message=".*cannot import name '_C'.*")
warnings.filterwarnings("ignore", message=".*Skipping the post-processing step.*")

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import rasterio

# Paths
EXP_DIR = Path(__file__).parent
OUTPUT_DIR = EXP_DIR / "output"
CONFIG_FILE = EXP_DIR / "config.yaml"

# Source image
SOURCE_IMAGE = Path("D:/TS_ann/data/Plant23_NDVI_MNF.tif")


def run_pipeline():
    """Run the segmentation pipeline."""
    from sam_mosaic import load_config, Pipeline

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load config
    print(f"\nConfig: {CONFIG_FILE}")
    config = load_config(CONFIG_FILE)

    # Get image info
    with rasterio.open(SOURCE_IMAGE) as src:
        width, height = src.width, src.height
        n_cols = (width + config.tiling.tile_size - 1) // config.tiling.tile_size
        n_rows = (height + config.tiling.tile_size - 1) // config.tiling.tile_size

    # Print experiment info
    print("\n" + "=" * 60)
    print("EXPERIMENT: Full Plant23 Image - V2 (Fixed Grid Borders)")
    print("=" * 60)
    print(f"Image: {width} x {height} pixels ({width * height:,} total)")
    print(f"Tile size: {config.tiling.tile_size} x {config.tiling.tile_size}")
    print(f"Grid: {n_cols} x {n_rows} = {n_cols * n_rows} base tiles")
    print(f"Discontinuities: {n_cols-1} V-lines, {n_rows-1} H-lines, {(n_cols-1)*(n_rows-1)} corners")
    print()
    print("Cascade config:")
    print(f"  Base tiles: {config.base_tiles.cascade.n_passes} passes, "
          f"{config.base_tiles.grid.points_per_side}x{config.base_tiles.grid.points_per_side} = "
          f"{config.base_tiles.grid.points_per_side**2} points (pass 0), K-means {config.base_tiles.cascade.points_per_pass} pts (pass 1+)")
    print(f"  V/H tiles: {config.border_correction.v_tiles.cascade.n_passes} passes, "
          f"{config.border_correction.v_tiles.grid.n_across}x{config.border_correction.v_tiles.grid.n_along} = "
          f"{config.border_correction.v_tiles.grid.n_across * config.border_correction.v_tiles.grid.n_along} points (pass 0), "
          f"K-means {config.border_correction.v_tiles.cascade.points_per_pass} pts in zone (pass 1+)")
    print(f"  Corners: {config.border_correction.corner_tiles.cascade.n_passes} passes, "
          f"{config.border_correction.corner_tiles.grid.n_x}x{config.border_correction.corner_tiles.grid.n_y} = "
          f"{config.border_correction.corner_tiles.grid.n_x * config.border_correction.corner_tiles.grid.n_y} points (pass 0), "
          f"K-means {config.border_correction.corner_tiles.cascade.points_per_pass} pts in zone (pass 1+)")
    print()
    print("Thresholds:")
    print(f"  Base: IoU & Stability {config.base_tiles.cascade.thresholds.iou[0]:.2f} -> "
          f"{config.base_tiles.cascade.thresholds.iou[1]:.2f}")
    print(f"  V/H/Corners: IoU & Stability {config.border_correction.v_tiles.cascade.thresholds.iou[0]:.2f} -> "
          f"{config.border_correction.v_tiles.cascade.thresholds.iou[1]:.2f}")
    print()
    pp = config.base_tiles.postprocess
    if pp.enabled:
        print("Post-processing (base tiles):")
        print(f"  Remove masks < {pp.min_area} px")
        print(f"  Merge enclosed < {pp.max_enclosed_area} px")
        print(f"  Edge completion: {pp.edge_max_distance} px search")
    print("=" * 60 + "\n")

    # Run
    start = time.time()
    pipeline = Pipeline(config)
    result = pipeline.run(SOURCE_IMAGE, OUTPUT_DIR)
    elapsed = time.time() - start

    # Results
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Output: {result}")

    # Coverage stats
    import numpy as np
    with rasterio.open(result) as src:
        # Read in chunks to avoid memory issues
        total = 0
        covered = 0
        labels_set = set()

        for ji, window in src.block_windows(1):
            data = src.read(1, window=window)
            total += data.size
            covered += (data > 0).sum()
            labels_set.update(np.unique(data))

        labels_set.discard(0)
        coverage = covered / total * 100
        n_instances = len(labels_set)

        print(f"\n{'=' * 60}")
        print("RESULTS")
        print("=" * 60)
        print(f"Coverage: {coverage:.2f}% ({covered:,} / {total:,} pixels)")
        print(f"Instances: {n_instances:,}")
        print("=" * 60)


def main():
    if not SOURCE_IMAGE.exists():
        print(f"ERROR: Source image not found: {SOURCE_IMAGE}")
        return

    print(f"Source: {SOURCE_IMAGE}")
    run_pipeline()


if __name__ == "__main__":
    main()
