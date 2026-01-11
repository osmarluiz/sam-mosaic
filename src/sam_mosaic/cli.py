"""Command-line interface for SAM-Mosaic."""

import argparse
import sys
from pathlib import Path

from sam_mosaic import __version__


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SAM-Mosaic: Large-scale image segmentation using SAM2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  sam-mosaic input.tif output/ --checkpoint path/to/sam2.pt

  # With config file
  sam-mosaic input.tif output/ --config config.yaml

  # Single-pass ablation
  sam-mosaic input.tif output/ --checkpoint sam2.pt \\
      --max-passes 1 --iou-start 0.86 --no-adaptive-threshold

  # No black mask ablation
  sam-mosaic input.tif output/ --checkpoint sam2.pt --no-black-mask

  # Custom tile size and padding
  sam-mosaic input.tif output/ --checkpoint sam2.pt \\
      --tile-size 2000 --padding 100
        """
    )

    # Required arguments
    parser.add_argument(
        "input",
        type=str,
        help="Input image path (GeoTIFF or similar)"
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output directory"
    )

    # Config options
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to SAM2 checkpoint file"
    )

    # Tile parameters
    tile_group = parser.add_argument_group("Tile parameters")
    tile_group.add_argument(
        "--tile-size",
        type=int,
        help="Tile size in pixels (default: 1000)"
    )
    tile_group.add_argument(
        "--padding",
        type=int,
        help="Padding pixels per side (default: 50)"
    )

    # Threshold parameters
    threshold_group = parser.add_argument_group("Threshold parameters")
    threshold_group.add_argument(
        "--iou-start",
        type=float,
        help="Initial IoU threshold (default: 0.93)"
    )
    threshold_group.add_argument(
        "--iou-end",
        type=float,
        help="Final IoU threshold (default: 0.60)"
    )
    threshold_group.add_argument(
        "--threshold-step",
        type=float,
        help="Threshold decrease per pass (default: 0.01)"
    )

    # Segmentation parameters
    seg_group = parser.add_argument_group("Segmentation parameters")
    seg_group.add_argument(
        "--points-per-side",
        type=int,
        help="Grid density per side (default: 64 = 4096 points)"
    )
    seg_group.add_argument(
        "--target-coverage",
        type=float,
        help="Stop at this coverage %% (default: 99.0)"
    )
    seg_group.add_argument(
        "--max-passes",
        type=int,
        help="Maximum passes per tile (default: unlimited)"
    )
    seg_group.add_argument(
        "--no-black-mask",
        action="store_true",
        help="Disable black masking of segmented areas"
    )
    seg_group.add_argument(
        "--no-adaptive-threshold",
        action="store_true",
        help="Use fixed threshold instead of adaptive"
    )

    # Merge parameters
    merge_group = parser.add_argument_group("Merge parameters")
    merge_group.add_argument(
        "--min-contact",
        type=int,
        help="Minimum contact pixels for merge (default: 5)"
    )
    merge_group.add_argument(
        "--min-area",
        type=int,
        help="Minimum region area to keep (default: 100)"
    )

    # Output options
    output_group = parser.add_argument_group("Output options")
    output_group.add_argument(
        "--no-shapefile",
        action="store_true",
        help="Don't save shapefile output"
    )
    output_group.add_argument(
        "--geopackage",
        action="store_true",
        help="Also save GeoPackage output"
    )
    output_group.add_argument(
        "--simplify-tolerance",
        type=float,
        help="Polygon simplification tolerance in map units (default: 1.0). Use 0 for no simplification."
    )

    # Other options
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"sam-mosaic {__version__}"
    )

    args = parser.parse_args()

    # Validate inputs
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Determine checkpoint and config
    checkpoint = args.checkpoint
    config_path = args.config

    # If no config or checkpoint specified, try to use default config
    if checkpoint is None and config_path is None:
        # Look for default config in package directory
        import sam_mosaic
        package_dir = Path(sam_mosaic.__file__).parent.parent.parent
        default_config = package_dir / "config" / "default.yaml"

        if default_config.exists():
            config_path = str(default_config)
            print(f"Using default config: {default_config}")
        else:
            print("Error: Either --checkpoint or --config must be specified", file=sys.stderr)
            print("       Or place config/default.yaml in the package directory", file=sys.stderr)
            sys.exit(1)

    # Build parameters dict for overrides
    params = {}

    if args.tile_size is not None:
        params["tile_size"] = args.tile_size
    if args.padding is not None:
        params["padding"] = args.padding
    if args.iou_start is not None:
        params["iou_start"] = args.iou_start
    if args.iou_end is not None:
        params["iou_end"] = args.iou_end
    if args.threshold_step is not None:
        params["threshold_step"] = args.threshold_step
    if args.points_per_side is not None:
        params["points_per_side"] = args.points_per_side
    if args.target_coverage is not None:
        params["target_coverage"] = args.target_coverage
    if args.max_passes is not None:
        params["max_passes"] = args.max_passes
    if args.no_black_mask:
        params["use_black_mask"] = False
    if args.no_adaptive_threshold:
        params["use_adaptive_threshold"] = False
    if args.min_contact is not None:
        params["min_contact_pixels"] = args.min_contact
    if args.min_area is not None:
        params["min_mask_area"] = args.min_area
    if args.no_shapefile:
        params["save_shapefile"] = False
    if args.geopackage:
        params["save_geopackage"] = True
    if args.simplify_tolerance is not None:
        params["simplify_tolerance"] = args.simplify_tolerance

    # Run segmentation
    try:
        if config_path:
            # Load config and apply overrides
            from sam_mosaic.config import load_config, Config
            config = load_config(config_path)

            if checkpoint:
                config.sam_checkpoint = checkpoint

            # Apply parameter overrides
            if params:
                config = Config.with_overrides(config, **params)

            from sam_mosaic.api import segment_image
            result = segment_image(
                args.input,
                args.output,
                config=config,
                verbose=not args.quiet
            )
        else:
            # Use direct parameters
            from sam_mosaic.api import segment_with_params
            result = segment_with_params(
                args.input,
                args.output,
                checkpoint=checkpoint,
                verbose=not args.quiet,
                **params
            )

        # Print summary
        if not args.quiet:
            print(f"\nSegmentation complete!")
            print(f"  Segments: {result.n_segments}")
            print(f"  Coverage: {result.coverage:.1f}%")
            print(f"  Time: {result.processing_time:.1f}s")

            if result.labels_path:
                print(f"  Labels: {result.labels_path}")
            if result.shapefile_path:
                print(f"  Shapefile: {result.shapefile_path}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
