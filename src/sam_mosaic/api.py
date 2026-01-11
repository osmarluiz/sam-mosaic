"""High-level API for SAM-Mosaic segmentation.

This module provides simple entry points for common use cases:
- segment_image(): Use with config file or defaults
- segment_with_params(): Direct parameter control for ablations
"""

from pathlib import Path
from typing import Optional, Union

from sam_mosaic.config import Config, load_config
from sam_mosaic.core.pipeline import Pipeline, SegmentationResult


def segment_image(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    config: Optional[Union[Config, str, Path]] = None,
    checkpoint: Optional[str] = None,
    verbose: bool = True
) -> SegmentationResult:
    """Segment a large image using SAM-Mosaic.

    This is the main entry point for standard usage. It handles
    configuration loading and pipeline execution.

    Args:
        input_path: Path to input image (GeoTIFF or similar).
        output_dir: Directory for output files.
        config: Configuration object, path to YAML file, or None for defaults.
        checkpoint: Path to SAM2 checkpoint (overrides config).
        verbose: Whether to print progress messages.

    Returns:
        SegmentationResult with output paths and statistics.

    Example:
        >>> from sam_mosaic import segment_image
        >>> result = segment_image(
        ...     "input.tif",
        ...     "output/",
        ...     checkpoint="path/to/sam2.pt"
        ... )
        >>> print(f"Segments: {result.n_segments}")
    """
    # Load or create config
    if config is None:
        cfg = Config()
    elif isinstance(config, (str, Path)):
        cfg = load_config(config)
    else:
        cfg = config

    # Override checkpoint if provided
    if checkpoint is not None:
        cfg.sam_checkpoint = checkpoint

    # Run pipeline
    pipeline = Pipeline(cfg)
    return pipeline.run(input_path, output_dir, verbose=verbose)


def segment_with_params(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    checkpoint: str,
    # Tile parameters
    tile_size: int = 1000,
    padding: int = 50,
    # Threshold parameters
    iou_start: float = 0.93,
    iou_end: float = 0.60,
    stability_start: float = 0.93,
    stability_end: float = 0.60,
    threshold_step: float = 0.01,
    # Segmentation parameters
    points_per_side: int = 64,
    target_coverage: float = 99.0,
    max_passes: Optional[int] = None,
    use_black_mask: bool = True,
    use_adaptive_threshold: bool = True,
    # Merge parameters
    min_contact_pixels: int = 5,
    min_mask_area: int = 100,
    merge_enclosed_max_area: int = 500,
    # Output parameters
    save_labels: bool = True,
    save_shapefile: bool = True,
    save_geopackage: bool = False,
    simplify_tolerance: float = 1.0,
    verbose: bool = True
) -> SegmentationResult:
    """Segment with direct parameter control.

    This function is designed for ablation studies and experiments
    where you need fine-grained control over all parameters.

    Args:
        input_path: Path to input image.
        output_dir: Directory for output files.
        checkpoint: Path to SAM2 checkpoint (required).

        tile_size: Tile size in pixels (default 1000).
        padding: Padding pixels per side (default 50).

        iou_start: Initial IoU threshold (default 0.93).
        iou_end: Final IoU threshold (default 0.60).
        stability_start: Initial stability threshold (default 0.93).
        stability_end: Final stability threshold (default 0.60).
        threshold_step: Threshold decrease per pass (default 0.01).

        points_per_side: Grid density (default 64 = 4096 points).
        target_coverage: Stop at this coverage % (default 99.0).
        max_passes: Maximum passes per tile (default None = unlimited).
        use_black_mask: Whether to mask segmented areas (default True).
        use_adaptive_threshold: Whether to decrease threshold (default True).

        min_contact_pixels: Minimum contact for merge (default 5).
        min_mask_area: Minimum region area to keep (default 100).
        merge_enclosed_max_area: Max area for enclosed merge (default 500).

        save_labels: Save label raster (default True).
        save_shapefile: Save shapefile (default True).
        save_geopackage: Save geopackage (default False).
        simplify_tolerance: Polygon simplification in map units (default 1.0). Use 0 for no simplification.

        verbose: Print progress (default True).

    Returns:
        SegmentationResult with output paths and statistics.

    Example:
        >>> # Single-pass ablation
        >>> result = segment_with_params(
        ...     "input.tif", "output/",
        ...     checkpoint="sam2.pt",
        ...     max_passes=1,
        ...     iou_start=0.86,
        ...     use_adaptive_threshold=False
        ... )

        >>> # No black mask ablation
        >>> result = segment_with_params(
        ...     "input.tif", "output/",
        ...     checkpoint="sam2.pt",
        ...     use_black_mask=False
        ... )
    """
    from sam_mosaic.config import (
        TileConfig, ThresholdConfig, SegmentationConfig,
        MergeConfig, OutputConfig
    )

    # Build config from parameters
    cfg = Config(
        tile=TileConfig(size=tile_size, padding=padding),
        threshold=ThresholdConfig(
            iou_start=iou_start,
            iou_end=iou_end,
            stability_start=stability_start,
            stability_end=stability_end,
            step=threshold_step,
        ),
        segmentation=SegmentationConfig(
            points_per_side=points_per_side,
            target_coverage=target_coverage,
            max_passes=max_passes,
            use_black_mask=use_black_mask,
            use_adaptive_threshold=use_adaptive_threshold,
        ),
        merge=MergeConfig(
            min_contact_pixels=min_contact_pixels,
            min_mask_area=min_mask_area,
            merge_enclosed_max_area=merge_enclosed_max_area,
        ),
        output=OutputConfig(
            save_labels=save_labels,
            save_shapefile=save_shapefile,
            save_geopackage=save_geopackage,
            simplify_tolerance=simplify_tolerance,
        ),
        sam_checkpoint=checkpoint,
    )

    # Run pipeline
    pipeline = Pipeline(cfg)
    return pipeline.run(input_path, output_dir, verbose=verbose)
