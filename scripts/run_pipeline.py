#!/usr/bin/env python3
"""Run the SAM-Mosaic segmentation pipeline."""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sam_mosaic import load_config, Pipeline


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def main():
    parser = argparse.ArgumentParser(
        description="SAM-Mosaic: Segment large images with SAM"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input image path (GeoTIFF or other GDAL-supported format)"
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output directory"
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        required=True,
        help="Configuration YAML file"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Validate inputs
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    # Load config
    logger.info(f"Loading config: {args.config}")
    config = load_config(args.config)

    # Run pipeline
    logger.info(f"Processing: {args.input}")
    pipeline = Pipeline(config)

    try:
        result = pipeline.run(args.input, args.output)
        logger.info(f"Done! Output: {result}")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if args.verbose:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
