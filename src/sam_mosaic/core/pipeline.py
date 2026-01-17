"""Main segmentation pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
import time
import gc
import numpy as np
import torch

from sam_mosaic.config import Config
from sam_mosaic.sam import SAMPredictor
from sam_mosaic.io import get_image_metadata, load_tile, save_labels, ImageMetadata
from sam_mosaic.merge import merge_at_discontinuities, merge_enclosed_and_remove_small
from sam_mosaic.core.tile import process_tile, calculate_grid_dimensions
from sam_mosaic.core.mosaic import create_mosaic_writer


@dataclass
class SegmentationResult:
    """Result from pipeline segmentation.

    Attributes:
        labels_path: Path to saved labels TIFF.
        shapefile_path: Path to saved shapefile (if enabled).
        stats_path: Path to saved stats JSON (if enabled).
        n_segments: Number of segments.
        coverage: Total coverage percentage.
        processing_time: Total processing time in seconds.
        tile_stats: Statistics per tile.
        merge_stats: Statistics from discontinuity merge step.
    """
    labels_path: Optional[Path] = None
    shapefile_path: Optional[Path] = None
    stats_path: Optional[Path] = None
    n_segments: int = 0
    coverage: float = 0.0
    processing_time: float = 0.0
    tile_stats: list = field(default_factory=list)
    merge_stats: dict = field(default_factory=dict)


class Pipeline:
    """Main segmentation pipeline.

    Orchestrates the full segmentation process:
    1. Load image metadata
    2. Process tiles with multi-pass segmentation
    3. Post-process (remove small, merge enclosed)
    4. Merge at tile discontinuities
    5. Vectorize and save outputs

    Example:
        >>> from sam_mosaic import Pipeline, Config
        >>> config = Config(sam_checkpoint="path/to/sam2.pt")
        >>> pipeline = Pipeline(config)
        >>> result = pipeline.run("input.tif", "output/")
    """

    def __init__(self, config: Config):
        """Initialize pipeline.

        Args:
            config: Configuration object.
        """
        self.config = config
        self._predictor: Optional[SAMPredictor] = None
        self._metadata: Optional[ImageMetadata] = None

    def run(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        verbose: bool = True
    ) -> SegmentationResult:
        """Run the full segmentation pipeline.

        Args:
            input_path: Path to input image.
            output_dir: Directory for output files.
            verbose: Whether to print progress.

        Returns:
            SegmentationResult with paths and statistics.
        """
        start_time = time.time()
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Validate config
        self.config.validate()
        if self.config.sam_checkpoint is None:
            raise ValueError("sam_checkpoint must be specified in config")

        # Get image metadata
        self._metadata = get_image_metadata(input_path)

        # Calculate grid
        tile_size = self.config.tile.size
        n_cols, n_rows, total_tiles = calculate_grid_dimensions(
            self._metadata.width,
            self._metadata.height,
            tile_size
        )

        if verbose:
            print("=" * 60)
            print("SAM-MOSAIC SEGMENTATION")
            print("=" * 60)
            print(f"  Input:  {input_path}")
            print(f"  Output: {output_dir}")
            print(f"  Image:  {self._metadata.width} x {self._metadata.height}")
            print(f"  Tiles:  {n_cols} x {n_rows} = {total_tiles}")
            print("-" * 60)

        # Check for debug mode via environment variable
        import os
        debug_mode = os.environ.get("SAM_MOSAIC_DEBUG", "").lower() in ("1", "true", "yes")

        if verbose:
            if debug_mode:
                print("Loading SAM2 model (DEBUG MODE)...")
            else:
                print("Loading SAM2 model...", end=" ", flush=True)

        # Initialize predictor
        self._predictor = SAMPredictor(self.config.sam_checkpoint)
        self._predictor.load_model(debug=debug_mode)

        if verbose and not debug_mode:
            print("OK")

        if debug_mode:
            import subprocess
            def _get_vram():
                try:
                    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                       capture_output=True, text=True, timeout=5)
                    return int(r.stdout.strip()) if r.returncode == 0 else 0
                except:
                    return 0
            print(f"[DEBUG] VRAM after model load: {_get_vram()} MB", flush=True)

        # Initialize mosaic writer
        mosaic_height = n_rows * tile_size
        mosaic_width = n_cols * tile_size
        streaming_mode = self.config.output.streaming_mode

        if debug_mode:
            print(f"[DEBUG] Creating mosaic writer ({streaming_mode} mode)...", flush=True)
            print(f"[DEBUG] Mosaic size: {mosaic_width} x {mosaic_height} = {mosaic_width*mosaic_height/1e9:.2f} Gpixels", flush=True)

        mosaic_writer = create_mosaic_writer(
            height=mosaic_height,
            width=mosaic_width,
            tile_size=tile_size,
            streaming_mode=streaming_mode,
            crs=self._metadata.crs,
            transform=self._metadata.transform
        )

        # Determine actual mode used (for verbose output)
        actual_mode = "disk" if type(mosaic_writer).__name__ == "DiskMosaic" else "ram"

        if debug_mode:
            print(f"[DEBUG] Mosaic writer created ({actual_mode})", flush=True)
            print(f"[DEBUG] VRAM after mosaic writer: {_get_vram()} MB", flush=True)

        if verbose and streaming_mode == "auto":
            print(f"  Mosaic mode: {actual_mode} (auto-selected)")

        # Process tiles
        if debug_mode:
            print(f"[DEBUG] About to start processing tiles...", flush=True)
            print(f"[DEBUG] VRAM before tile loop: {_get_vram()} MB", flush=True)

        if verbose:
            print("-" * 60)
            print(f"Processing {total_tiles} tiles...")
            print("-" * 60)

        label_offset = 0
        tile_stats = []
        tile_start_time = time.time()

        if debug_mode:
            print(f"[DEBUG] Starting tile loop...", flush=True)

        for row in range(n_rows):
            for col in range(n_cols):
                if debug_mode and row == 0 and col == 0:
                    print(f"[DEBUG] Loading first tile...", flush=True)
                    print(f"[DEBUG] VRAM before first tile: {_get_vram()} MB", flush=True)
                tile_idx = row * n_cols + col + 1

                # Load tile
                tile_info = load_tile(
                    input_path,
                    row=row,
                    col=col,
                    tile_size=tile_size,
                    padding=self.config.tile.padding
                )

                # Process tile
                result = process_tile(
                    self._predictor,
                    tile_info,
                    self.config,
                    start_label=label_offset + 1
                )

                # Write tile to mosaic
                mosaic_writer.write_tile(result.labels, row, col)
                label_offset = int(result.labels.max()) if result.labels.max() > 0 else label_offset

                tile_stats.append({
                    "row": row,
                    "col": col,
                    "coverage": result.coverage,
                    "n_labels": result.n_labels,
                    "n_passes": result.n_passes,
                    "masks_per_pass": result.pass_stats.get("masks_per_pass", []),
                    "coverage_per_pass": result.pass_stats.get("coverage_per_pass", []),
                    "final_iou": result.pass_stats.get("final_iou", 0.0),
                    # Detailed stats for ablation analysis
                    "masks_generated_per_pass": result.pass_stats.get("masks_generated_per_pass", []),
                    "masks_filtered_overlap": result.pass_stats.get("masks_filtered_overlap", 0),
                    "masks_filtered_small": result.pass_stats.get("masks_filtered_small", 0),
                })

                if verbose:
                    elapsed = time.time() - tile_start_time
                    avg_time = elapsed / tile_idx
                    eta = avg_time * (total_tiles - tile_idx)
                    eta_str = f"{eta/60:.1f}min" if eta >= 60 else f"{eta:.0f}s"

                    print(f"  Tile {tile_idx:3d}/{total_tiles} [{row},{col}] | "
                          f"{result.coverage:5.1f}% | {result.n_labels:4d} seg | "
                          f"{result.n_passes} passes | ETA: {eta_str}")

                # Cleanup: reset predictor state for next tile
                self._predictor.reset_image()

                # Batch GPU memory cleanup (every 10 tiles) to reduce sync overhead
                if tile_idx % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

        # Final GPU cleanup after all tiles
        gc.collect()
        torch.cuda.empty_cache()

        # Post-processing
        if verbose:
            print("-" * 60)
            print("Post-processing...", flush=True)

        # Load mosaic into memory for post-processing
        mosaic = mosaic_writer.read()

        # Merge enclosed regions + remove small (combined for efficiency - single ndimage.label)
        t0 = time.time()
        mosaic = merge_enclosed_and_remove_small(
            mosaic,
            merge_enclosed_max_area=self.config.merge.merge_enclosed_max_area,
            remove_small_min_area=self.config.merge.min_mask_area
        )
        if verbose:
            print(f"  Merge enclosed + remove small: done ({time.time()-t0:.1f}s)", flush=True)

        # Calculate segment counts BEFORE merge
        segment_counts_before = np.bincount(mosaic.ravel())

        # Only copy mosaic if we need to save labels_before_merge.tif
        # (This avoids 400MB+ copy when not saving intermediate labels)
        mosaic_before_merge = mosaic.copy() if self.config.output.save_labels else None

        # Merge at discontinuities
        t0 = time.time()
        mosaic, merge_stats = merge_at_discontinuities(
            mosaic,
            tile_size,
            self.config.merge.min_contact_pixels,
            return_stats=True
        )
        if verbose:
            print(f"  Merge at edges: done ({time.time()-t0:.1f}s) "
                  f"[{merge_stats['labels_merged']} labels merged]", flush=True)

        # Calculate final stats (segment_counts_before already computed above)
        segment_counts = np.bincount(mosaic.ravel())

        # Segment counts (exclude background at index 0)
        n_segments = int((segment_counts[1:] > 0).sum()) if len(segment_counts) > 1 else 0
        n_segments_before_merge = int((segment_counts_before[1:] > 0).sum()) if len(segment_counts_before) > 1 else 0

        # Coverage from the same bincount
        coverage = (mosaic.size - segment_counts[0]) / mosaic.size * 100

        # Merge stats
        merge_stats["segments_before_merge"] = n_segments_before_merge
        merge_stats["segments_after_merge"] = n_segments
        merge_stats["segments_reduced"] = n_segments_before_merge - n_segments

        # Segment size statistics (reuse segment_counts, exclude background)
        segment_areas = segment_counts[1:]  # Exclude background (index 0)
        segment_areas = segment_areas[segment_areas > 0]  # Remove empty labels
        segment_size_stats = {
            "mean": float(np.mean(segment_areas)) if len(segment_areas) > 0 else 0,
            "median": float(np.median(segment_areas)) if len(segment_areas) > 0 else 0,
            "std": float(np.std(segment_areas)) if len(segment_areas) > 0 else 0,
            "min": int(np.min(segment_areas)) if len(segment_areas) > 0 else 0,
            "max": int(np.max(segment_areas)) if len(segment_areas) > 0 else 0,
        }

        # Threshold distribution (count tiles at each final IoU)
        from collections import Counter
        final_ious = [ts.get("final_iou", 0) for ts in tile_stats]
        iou_distribution = dict(Counter([round(iou, 2) for iou in final_ious]))

        # Save outputs
        result = SegmentationResult(
            n_segments=n_segments,
            coverage=coverage,
            processing_time=time.time() - start_time,
            tile_stats=tile_stats,
            merge_stats=merge_stats,
        )

        if verbose:
            print("-" * 60)
            print("Saving outputs...")

        if self.config.output.save_labels:
            # Save intermediate (before merge at edges)
            labels_before_merge_path = output_dir / "labels_before_merge.tif"
            save_labels(
                mosaic_before_merge,
                labels_before_merge_path,
                crs=self._metadata.crs,
                transform=self._metadata.transform
            )
            if verbose:
                print(f"  Labels (before merge): {labels_before_merge_path}")

            # Save final (after merge at edges)
            labels_path = output_dir / "labels.tif"
            save_labels(
                mosaic,
                labels_path,
                crs=self._metadata.crs,
                transform=self._metadata.transform
            )
            result.labels_path = labels_path
            if verbose:
                print(f"  Labels (final):        {labels_path}")

        # Vectorize once, save to multiple formats if needed
        if self.config.output.save_shapefile or self.config.output.save_geopackage:
            from sam_mosaic.vectorize.polygonize import extract_polygons, save_shapefile, save_geopackage

            # Extract polygons once (expensive operation)
            features = extract_polygons(
                mosaic,
                transform=self._metadata.transform,
                simplify_tolerance=self.config.output.simplify_tolerance
            )

            if self.config.output.save_shapefile:
                shapefile_path = output_dir / "segments.shp"
                save_shapefile(features, shapefile_path, crs=self._metadata.crs)
                result.shapefile_path = shapefile_path
                if verbose:
                    print(f"  Shapefile: {shapefile_path}")

            if self.config.output.save_geopackage:
                geopackage_path = output_dir / "segments.gpkg"
                save_geopackage(features, geopackage_path, crs=self._metadata.crs)
                if verbose:
                    print(f"  GeoPackage: {geopackage_path}")

        if self.config.output.save_stats:
            import json
            stats_path = output_dir / "stats.json"
            stats_data = {
                "summary": {
                    "input_path": str(input_path),
                    "image_width": self._metadata.width,
                    "image_height": self._metadata.height,
                    "tile_size": tile_size,
                    "n_cols": n_cols,
                    "n_rows": n_rows,
                    "total_tiles": total_tiles,
                    "total_segments": n_segments,
                    "total_coverage": round(coverage, 2),
                    "processing_time_seconds": round(result.processing_time, 2),
                },
                "segment_sizes": segment_size_stats,
                "threshold_distribution": iou_distribution,
                "config": {
                    "tile": {"size": self.config.tile.size, "padding": self.config.tile.padding},
                    "threshold": {
                        "iou_start": self.config.threshold.iou_start,
                        "iou_end": self.config.threshold.iou_end,
                        "stability_start": self.config.threshold.stability_start,
                        "stability_end": self.config.threshold.stability_end,
                        "step": self.config.threshold.step,
                    },
                    "segmentation": {
                        "points_per_side": self.config.segmentation.points_per_side,
                        "target_coverage": self.config.segmentation.target_coverage,
                        "max_passes": self.config.segmentation.max_passes,
                        "use_black_mask": self.config.segmentation.use_black_mask,
                        "use_adaptive_threshold": self.config.segmentation.use_adaptive_threshold,
                        "point_strategy": self.config.segmentation.point_strategy,
                        "erosion_iterations": self.config.segmentation.erosion_iterations,
                    },
                    "merge": {
                        "min_contact_pixels": self.config.merge.min_contact_pixels,
                        "min_mask_area": self.config.merge.min_mask_area,
                        "merge_enclosed_max_area": self.config.merge.merge_enclosed_max_area,
                    },
                },
                "merge_stats": merge_stats,
                "tiles": tile_stats,
            }
            with open(stats_path, "w") as f:
                json.dump(stats_data, f, indent=2)
            result.stats_path = stats_path
            if verbose:
                print(f"  Stats:     {stats_path}")

        # Cleanup
        mosaic_writer.close()
        self._predictor.unload_model()
        self._predictor = None

        if verbose:
            total_time = result.processing_time
            time_str = f"{total_time/60:.1f} min" if total_time >= 60 else f"{total_time:.1f}s"
            print("=" * 60)
            print("COMPLETE")
            print("=" * 60)
            print(f"  Segments: {n_segments:,}")
            print(f"  Coverage: {coverage:.1f}%")
            print(f"  Time:     {time_str}")
            print("=" * 60)

        return result
