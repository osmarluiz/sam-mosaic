"""Ablation study example for SAM-Mosaic.

This script demonstrates how to run ablation experiments
to validate each component of the methodology.
"""

from pathlib import Path
from sam_mosaic import segment_with_params

# Configuration
INPUT_IMAGE = "path/to/input.tif"
OUTPUT_BASE = Path("ablation_results")
CHECKPOINT = "path/to/sam2.1_hiera_large.pt"


def run_ablations():
    """Run all ablation experiments."""

    # ==========================================================
    # PART 1: Single-pass is not sufficient
    # ==========================================================

    # Single-pass with SAM default threshold (0.86)
    print("\n" + "="*60)
    print("Ablation: Single-pass with default threshold (0.86)")
    print("="*60)
    result = segment_with_params(
        INPUT_IMAGE,
        OUTPUT_BASE / "single_pass_default",
        checkpoint=CHECKPOINT,
        max_passes=1,
        iou_start=0.86,
        use_adaptive_threshold=False
    )
    print(f"Coverage: {result.coverage:.1f}%")

    # Single-pass with high threshold (0.93)
    print("\n" + "="*60)
    print("Ablation: Single-pass with high threshold (0.93)")
    print("="*60)
    result = segment_with_params(
        INPUT_IMAGE,
        OUTPUT_BASE / "single_pass_high",
        checkpoint=CHECKPOINT,
        max_passes=1,
        iou_start=0.93,
        use_adaptive_threshold=False
    )
    print(f"Coverage: {result.coverage:.1f}%")

    # Single-pass with low threshold (0.70)
    print("\n" + "="*60)
    print("Ablation: Single-pass with low threshold (0.70)")
    print("="*60)
    result = segment_with_params(
        INPUT_IMAGE,
        OUTPUT_BASE / "single_pass_low",
        checkpoint=CHECKPOINT,
        max_passes=1,
        iou_start=0.70,
        use_adaptive_threshold=False
    )
    print(f"Coverage: {result.coverage:.1f}%")

    # ==========================================================
    # PART 2: Multi-pass with black mask
    # ==========================================================

    # Baseline (full methodology)
    print("\n" + "="*60)
    print("BASELINE: Full methodology")
    print("="*60)
    result = segment_with_params(
        INPUT_IMAGE,
        OUTPUT_BASE / "baseline",
        checkpoint=CHECKPOINT
        # All defaults: multi-pass, adaptive threshold, black mask
    )
    print(f"Coverage: {result.coverage:.1f}%")

    # Without black mask
    print("\n" + "="*60)
    print("Ablation: Multi-pass WITHOUT black mask")
    print("="*60)
    result = segment_with_params(
        INPUT_IMAGE,
        OUTPUT_BASE / "no_blackmask",
        checkpoint=CHECKPOINT,
        use_black_mask=False
    )
    print(f"Coverage: {result.coverage:.1f}%")

    # ==========================================================
    # PART 3: Adaptive threshold importance
    # ==========================================================

    # Multi-pass with fixed threshold
    print("\n" + "="*60)
    print("Ablation: Multi-pass with FIXED threshold (0.86)")
    print("="*60)
    result = segment_with_params(
        INPUT_IMAGE,
        OUTPUT_BASE / "fixed_threshold",
        checkpoint=CHECKPOINT,
        use_adaptive_threshold=False,
        iou_start=0.86
    )
    print(f"Coverage: {result.coverage:.1f}%")

    # ==========================================================
    # PART 4: Padding importance for merge
    # ==========================================================

    # Without padding
    print("\n" + "="*60)
    print("Ablation: Without padding (merge quality)")
    print("="*60)
    result = segment_with_params(
        INPUT_IMAGE,
        OUTPUT_BASE / "no_padding",
        checkpoint=CHECKPOINT,
        padding=0
    )
    print(f"Coverage: {result.coverage:.1f}%")

    print("\n" + "="*60)
    print("All ablations complete!")
    print("="*60)
    print(f"Results saved to: {OUTPUT_BASE.absolute()}")


if __name__ == "__main__":
    run_ablations()
