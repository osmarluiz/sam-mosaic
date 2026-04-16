"""Replicate SamGeo2 behavior: single-pass SAM2, no multi-pass, no padding.
Measures processing time for the three evaluation scenes.
"""
import time
import sys
sys.path.insert(0, "D:/projects/sam-mosaic2/src")

from sam_mosaic.api import segment_image
from sam_mosaic.config.schema import Config, TileConfig, SegmentationConfig, ThresholdConfig, MergeConfig

datasets = [
    ("bsb1", "D:/projects/sam-mosaic2/experiments/paper_jag/data/brasilia/bsb_1.tif", 250),
    ("potsdam1", "D:/projects/sam-mosaic2/experiments/paper_jag/data/potsdam/top_potsdam_3_13_RGB.tif", 1000),
    ("plant23", "D:/projects/sam-mosaic2/experiments/paper_jag/data/plant23/plant23_10k.tif", 1000),
]

print("Starting SamGeo2-equivalent timing runs...", flush=True)

for name, path, tile_size in datasets:
    print(f"\n{'='*60}", flush=True)
    print(f"  {name} (T={tile_size}, single-pass, no padding)", flush=True)
    print(f"{'='*60}", flush=True)

    config = Config(
        sam_checkpoint="D:/projects/sam-mosaic2/checkpoints/sam2.1_hiera_large.pt",
        tile=TileConfig(size=tile_size, padding=0),
        segmentation=SegmentationConfig(
            points_per_side=32,
            target_coverage=100.0,
            max_passes=1,
            use_black_mask=False,
            use_adaptive_threshold=False,
        ),
        threshold=ThresholdConfig(
            iou_start=0.88,
            iou_end=0.88,
            stability_start=0.95,
            stability_end=0.95,
            step=0.01,
        ),
        merge=MergeConfig(merge_strategy="none"),
    )

    out_dir = f"D:/projects/sam-mosaic2/experiments/paper_jag/output/v3/samgeo2_baseline/{name}_timing"

    t0 = time.time()
    try:
        result = segment_image(path, out_dir, config=config)
        elapsed = time.time() - t0
        print(f"  DONE: {elapsed:.0f}s", flush=True)
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  ERROR after {elapsed:.0f}s: {e}", flush=True)
        import traceback
        traceback.print_exc()

print("\nAll done!", flush=True)
