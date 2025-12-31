"""Main pipeline orchestrating the full segmentation workflow."""

from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging

import numpy as np
from tqdm import tqdm

from sam_mosaic.config import Config
from sam_mosaic.io.loader import load_image, get_metadata, load_tile, load_region
from sam_mosaic.io.writer import save_raster, save_labels
from sam_mosaic.tiling.grid import TileGrid
from sam_mosaic.tiling.borders import BorderTiles
from sam_mosaic.points.grids import (
    make_uniform_grid, make_v_grid, make_h_grid, make_corner_grid, filter_by_mask
)
from sam_mosaic.segmentation.sam import SAMPredictor
from sam_mosaic.segmentation.cascade import (
    run_cascade_on_tile, masks_to_labels,
    run_single_pass, save_combined_mask, load_combined_mask
)
from sam_mosaic.merge.bands import merge_bands, stitch_tile_labels
from sam_mosaic.vectorize.polygonize import masks_to_polygons, save_shapefile, save_geopackage


logger = logging.getLogger(__name__)


class Pipeline:
    """Full segmentation pipeline."""

    def __init__(self, config: Config):
        """Initialize pipeline.

        Args:
            config: Pipeline configuration.
        """
        self.config = config
        self.predictor: Optional[SAMPredictor] = None
        self._metadata = None

    def load_model(self) -> None:
        """Load SAM model."""
        logger.info(f"Loading SAM model: {self.config.sam.model_type}")
        self.predictor = SAMPredictor(
            checkpoint=self.config.sam.checkpoint,
            model_type=self.config.sam.model_type,
            device=self.config.sam.device
        )

    def run(
        self,
        input_path: str | Path,
        output_dir: str | Path
    ) -> Path:
        """Run full segmentation pipeline.

        Args:
            input_path: Path to input image.
            output_dir: Output directory.

        Returns:
            Path to final merged labels.
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load model if not already loaded
        if self.predictor is None:
            self.load_model()

        # Get image metadata
        self._metadata = get_metadata(input_path)
        logger.info(f"Image size: {self._metadata.width}x{self._metadata.height}")

        # Create tile grid
        grid = TileGrid(
            self._metadata.width,
            self._metadata.height,
            self.config.tiling.tile_size,
            self.config.tiling.overlap
        )
        logger.info(f"Tile grid: {grid.n_cols}x{grid.n_rows} = {len(grid)} tiles")

        # Stage 1: Base tiles cascade
        logger.info("Stage 1: Processing base tiles")
        band0, next_label = self._process_base_tiles(input_path, grid, output_dir)
        logger.info(f"Stage 1 complete: {next_label - 1} labels")

        # Stage 2: V/H border correction (continues label numbering)
        logger.info("Stage 2: Processing V/H borders")
        band1, next_label = self._process_vh_borders(input_path, grid, output_dir, start_label=next_label)
        logger.info(f"Stage 2 complete: labels now up to {next_label - 1}")

        # Stage 3: Corner correction (continues label numbering)
        logger.info("Stage 3: Processing corners")
        band2, next_label = self._process_corners(input_path, grid, output_dir, start_label=next_label)
        logger.info(f"Stage 3 complete: {next_label - 1} total labels")

        # Stage 4: Merge bands (simple overlay - IDs are already globally unique)
        logger.info("Stage 4: Merging bands")
        # Priority: corners (band2) > V/H (band1) > base (band0)
        merged = band0.copy()
        merged[band1 > 0] = band1[band1 > 0]  # V/H overwrites base
        merged[band2 > 0] = band2[band2 > 0]  # corners overwrite all

        # Save merged result
        merged_path = output_dir / "merged_labels.tif"
        save_labels(
            merged,
            merged_path,
            crs=self._metadata.crs,
            transform=self._metadata.transform
        )
        logger.info(f"Saved merged labels: {merged_path}")

        # Stage 5: Vectorize if requested
        if "shp" in self.config.output.formats:
            logger.info("Stage 5: Vectorizing to shapefile")
            features = masks_to_polygons(
                merged,
                transform=self._metadata.transform,
                crs=self._metadata.crs
            )
            shp_path = output_dir / "segments.shp"
            save_shapefile(features, shp_path, crs=self._metadata.crs)
            logger.info(f"Saved shapefile: {shp_path}")

        if "gpkg" in self.config.output.formats:
            features = masks_to_polygons(
                merged,
                transform=self._metadata.transform,
                crs=self._metadata.crs
            )
            gpkg_path = output_dir / "segments.gpkg"
            save_geopackage(features, gpkg_path, crs=self._metadata.crs)
            logger.info(f"Saved GeoPackage: {gpkg_path}")

        return merged_path

    def _process_base_tiles(
        self,
        input_path: Path,
        grid: TileGrid,
        output_dir: Path
    ) -> Tuple[np.ndarray, int]:
        """Process base tiles with cascade refinement (pass-by-pass).

        Processes all tiles for each pass before moving to the next pass.
        Converts masks to labels immediately with global unique IDs.

        Returns:
            Tuple of (band0 labels array, next available ID).
        """
        height = self._metadata.height
        width = self._metadata.width
        cfg = self.config.base_tiles

        # Directories
        state_dir = output_dir / "state" / "base"
        labels_dir = output_dir / "state" / "base_labels"
        state_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Pre-generate point grid (same for all tiles)
        tiles = list(grid)
        tile_size = self.config.tiling.tile_size
        base_points = make_uniform_grid(tile_size, tile_size, cfg.grid.points_per_side)
        logger.info(f"Base grid: {len(base_points)} points ({cfg.grid.points_per_side}x{cfg.grid.points_per_side})")

        # Global label counter (starts at 1, 0 = background)
        current_label = 1

        # Process pass by pass
        n_passes = cfg.cascade.n_passes

        for pass_idx in tqdm(range(n_passes), desc="Passes", unit="pass"):
            iou_thresh, _ = cfg.cascade.thresholds.interpolate(pass_idx, n_passes)
            total_coverage = 0.0
            pass_new_masks = 0

            tile_pbar = tqdm(tiles, desc=f"  Pass {pass_idx}", leave=False, unit="tile")

            for tile in tile_pbar:
                tile_id = f"tile_{tile.row}_{tile.col}"
                mask_path = state_dir / f"{tile_id}_mask.npy"
                labels_path = labels_dir / f"{tile_id}_labels.npy"

                # Load tile image
                tile_img = load_tile(
                    input_path,
                    tile.row, tile.col,
                    self.config.tiling.tile_size
                )

                # Load combined mask from disk (or create empty)
                combined_mask = load_combined_mask(mask_path, tile.height, tile.width)

                # Load labels from disk (or create empty)
                if labels_path.exists():
                    tile_labels = np.load(labels_path)
                else:
                    tile_labels = np.zeros((tile.height, tile.width), dtype=np.uint32)

                # Run single pass
                new_masks, combined_mask, coverage = run_single_pass(
                    predictor=self.predictor,
                    image=tile_img,
                    combined_mask=combined_mask,
                    initial_points=base_points,
                    config=cfg.cascade,
                    pass_idx=pass_idx,
                    mask_filter=None
                )

                # Convert new masks to labels IMMEDIATELY with global IDs
                # Only fill empty pixels (don't overwrite existing labels)
                for mask in new_masks:
                    tile_labels[(mask.mask > 0) & (tile_labels == 0)] = current_label
                    current_label += 1

                # Save state to disk
                save_combined_mask(combined_mask, mask_path)
                np.save(labels_path, tile_labels)

                total_coverage += coverage
                pass_new_masks += len(new_masks)
                tile_pbar.set_postfix({
                    "new": len(new_masks),
                    "cov": f"{coverage:.1f}%"
                })

            avg_coverage = total_coverage / len(tiles)
            logger.info(f"Pass {pass_idx}: iou={iou_thresh:.2f}, new_masks={pass_new_masks}, total_labels={current_label-1}, avg_cov={avg_coverage:.1f}%")

            # Save intermediate mosaic if requested
            if self.config.output.save_intermediate:
                self._save_labels_mosaic(tiles, labels_dir, output_dir, pass_idx, "base")

            # Early stop if all tiles converged
            if pass_idx > 0 and avg_coverage > 99.5:
                logger.info(f"Converged at pass {pass_idx}")
                break

        # Stitch all tile labels into band0
        band0 = np.zeros((height, width), dtype=np.uint32)
        for tile in tiles:
            tile_id = f"tile_{tile.row}_{tile.col}"
            labels_path = labels_dir / f"{tile_id}_labels.npy"
            tile_labels = np.load(labels_path)

            y_end = min(tile.y + tile_labels.shape[0], height)
            x_end = min(tile.x + tile_labels.shape[1], width)
            band0[tile.y:y_end, tile.x:x_end] = tile_labels[:y_end-tile.y, :x_end-tile.x]

        if self.config.output.save_intermediate:
            save_labels(
                band0,
                output_dir / "band0_base.tif",
                crs=self._metadata.crs,
                transform=self._metadata.transform
            )

        return band0, current_label

    def _save_pass_mosaic(
        self,
        tiles: list,
        state_dir: Path,
        output_dir: Path,
        pass_idx: int,
        stage: str
    ) -> None:
        """Save mosaic of combined masks after a pass."""
        from sam_mosaic.io.writer import save_mask

        height = self._metadata.height
        width = self._metadata.width
        mosaic = np.zeros((height, width), dtype=np.uint8)

        for tile in tiles:
            tile_id = f"tile_{tile.row}_{tile.col}"
            mask_path = state_dir / f"{tile_id}_mask.npy"
            if mask_path.exists():
                tile_mask = np.load(mask_path)
                y_end = min(tile.y + tile_mask.shape[0], height)
                x_end = min(tile.x + tile_mask.shape[1], width)
                mosaic[tile.y:y_end, tile.x:x_end] = tile_mask[:y_end-tile.y, :x_end-tile.x]

        passes_dir = output_dir / "passes" / stage
        passes_dir.mkdir(parents=True, exist_ok=True)
        save_mask(mosaic, passes_dir / f"mosaic_pass{pass_idx:02d}.tif")

    def _save_labels_mosaic(
        self,
        tiles: list,
        labels_dir: Path,
        output_dir: Path,
        pass_idx: int,
        stage: str
    ) -> None:
        """Save mosaic of labels after a pass."""
        height = self._metadata.height
        width = self._metadata.width
        mosaic = np.zeros((height, width), dtype=np.uint32)

        for tile in tiles:
            tile_id = f"tile_{tile.row}_{tile.col}"
            labels_path = labels_dir / f"{tile_id}_labels.npy"
            if labels_path.exists():
                tile_labels = np.load(labels_path)
                y_end = min(tile.y + tile_labels.shape[0], height)
                x_end = min(tile.x + tile_labels.shape[1], width)
                mosaic[tile.y:y_end, tile.x:x_end] = tile_labels[:y_end-tile.y, :x_end-tile.x]

        passes_dir = output_dir / "passes" / stage
        passes_dir.mkdir(parents=True, exist_ok=True)
        save_labels(
            mosaic,
            passes_dir / f"labels_pass{pass_idx:02d}.tif",
            crs=self._metadata.crs,
            transform=self._metadata.transform
        )

    def _process_vh_borders(
        self,
        input_path: Path,
        grid: TileGrid,
        output_dir: Path,
        start_label: int = 1
    ) -> Tuple[np.ndarray, int]:
        """Process V/H discontinuity tiles (pass-by-pass).

        Converts masks to labels immediately - no mask storage in memory.

        Args:
            start_label: Starting label ID (continues from base tiles).

        Returns:
            Tuple of (band1 labels array, next available ID).
        """
        from sam_mosaic.tiling.borders import crosses_v, crosses_h

        height = self._metadata.height
        width = self._metadata.width
        tile_size = self.config.tiling.tile_size
        zone_width = self.config.border_correction.zone_width
        current_label = start_label

        borders = BorderTiles(grid, zone_width=zone_width)

        # State directories
        state_dir_v = output_dir / "state" / "v"
        state_dir_h = output_dir / "state" / "h"
        state_dir_v.mkdir(parents=True, exist_ok=True)
        state_dir_h.mkdir(parents=True, exist_ok=True)

        # Get all tiles
        v_tiles = borders.get_v_tiles()
        h_tiles = borders.get_h_tiles()
        v_cfg = self.config.border_correction.v_tiles
        h_cfg = self.config.border_correction.h_tiles

        # Generate FIXED point grids (same for all tiles of each type)
        v_points = make_v_grid(
            tile_size=tile_size,
            zone_width=zone_width,
            n_across=v_cfg.grid.n_across,
            n_along=v_cfg.grid.n_along
        )
        h_points = make_h_grid(
            tile_size=tile_size,
            zone_width=zone_width,
            n_across=h_cfg.grid.n_across,
            n_along=h_cfg.grid.n_along
        )

        logger.info(f"V grid: {len(v_points)} points, H grid: {len(h_points)} points")

        # Output band - write labels directly here
        band1 = np.zeros((height, width), dtype=np.uint32)

        # Pre-generate tile info with filters (no masks list!)
        v_tile_info = []
        for i, btile in enumerate(v_tiles):
            v_tile_info.append({
                'id': f"v_{i}",
                'btile': btile,
                'filter': lambda m, bx=btile.x, dx=btile.disc_x: crosses_v(m.mask, dx, bx),
            })

        h_tile_info = []
        for i, btile in enumerate(h_tiles):
            h_tile_info.append({
                'id': f"h_{i}",
                'btile': btile,
                'filter': lambda m, by=btile.y, dy=btile.disc_y: crosses_h(m.mask, dy, by),
            })

        # Process V tiles pass by pass
        n_passes_v = v_cfg.cascade.n_passes
        logger.info(f"Processing {len(v_tiles)} V border tiles")

        for pass_idx in tqdm(range(n_passes_v), desc="V Passes", unit="pass"):
            pass_new = 0
            for info in tqdm(v_tile_info, desc=f"  V Pass {pass_idx}", leave=False, unit="tile"):
                btile = info['btile']
                tile_id = info['id']
                mask_path = state_dir_v / f"{tile_id}_mask.npy"

                region = load_region(input_path, btile.x, btile.y, btile.width, btile.height)
                combined_mask = load_combined_mask(mask_path, btile.height, btile.width)

                new_masks, combined_mask, coverage = run_single_pass(
                    predictor=self.predictor,
                    image=region,
                    combined_mask=combined_mask,
                    initial_points=v_points,
                    config=v_cfg.cascade,
                    pass_idx=pass_idx,
                    mask_filter=info['filter']
                )

                save_combined_mask(combined_mask, mask_path)

                # Convert masks to labels IMMEDIATELY
                # Only fill empty pixels (don't overwrite existing labels)
                # Skip masks that are too large (>40% of tile)
                tile_pixels = btile.width * btile.height
                max_mask_pixels = int(tile_pixels * 0.4)

                for mask in new_masks:
                    mask_size = mask.mask.sum()
                    if mask_size > max_mask_pixels:
                        continue  # Skip oversized masks

                    y_end = min(btile.y + mask.mask.shape[0], height)
                    x_end = min(btile.x + mask.mask.shape[1], width)
                    local_mask = mask.mask[:y_end - btile.y, :x_end - btile.x]
                    region = band1[btile.y:y_end, btile.x:x_end]
                    region[(local_mask > 0) & (region == 0)] = current_label
                    current_label += 1
                    pass_new += 1

            logger.debug(f"V Pass {pass_idx}: {pass_new} new masks")

        # Process H tiles pass by pass
        n_passes_h = h_cfg.cascade.n_passes
        logger.info(f"Processing {len(h_tiles)} H border tiles")

        for pass_idx in tqdm(range(n_passes_h), desc="H Passes", unit="pass"):
            pass_new = 0
            for info in tqdm(h_tile_info, desc=f"  H Pass {pass_idx}", leave=False, unit="tile"):
                btile = info['btile']
                tile_id = info['id']
                mask_path = state_dir_h / f"{tile_id}_mask.npy"

                region = load_region(input_path, btile.x, btile.y, btile.width, btile.height)
                combined_mask = load_combined_mask(mask_path, btile.height, btile.width)

                new_masks, combined_mask, coverage = run_single_pass(
                    predictor=self.predictor,
                    image=region,
                    combined_mask=combined_mask,
                    initial_points=h_points,
                    config=h_cfg.cascade,
                    pass_idx=pass_idx,
                    mask_filter=info['filter']
                )

                save_combined_mask(combined_mask, mask_path)

                # Convert masks to labels IMMEDIATELY
                # Only fill empty pixels (don't overwrite existing labels)
                # Skip masks that are too large (>40% of tile)
                tile_pixels = btile.width * btile.height
                max_mask_pixels = int(tile_pixels * 0.4)

                for mask in new_masks:
                    mask_size = mask.mask.sum()
                    if mask_size > max_mask_pixels:
                        continue  # Skip oversized masks

                    y_end = min(btile.y + mask.mask.shape[0], height)
                    x_end = min(btile.x + mask.mask.shape[1], width)
                    local_mask = mask.mask[:y_end - btile.y, :x_end - btile.x]
                    region = band1[btile.y:y_end, btile.x:x_end]
                    region[(local_mask > 0) & (region == 0)] = current_label
                    current_label += 1
                    pass_new += 1

            logger.debug(f"H Pass {pass_idx}: {pass_new} new masks")

        if self.config.output.save_intermediate:
            save_labels(
                band1,
                output_dir / "band1_vh.tif",
                crs=self._metadata.crs,
                transform=self._metadata.transform
            )

        return band1, current_label

    def _process_corners(
        self,
        input_path: Path,
        grid: TileGrid,
        output_dir: Path,
        start_label: int = 1
    ) -> Tuple[np.ndarray, int]:
        """Process corner tiles (pass-by-pass).

        Converts masks to labels immediately - no mask storage in memory.

        Args:
            start_label: Starting label ID (continues from V/H borders).

        Returns:
            Tuple of (band2 labels array, next available ID).
        """
        from sam_mosaic.tiling.borders import crosses_corner

        height = self._metadata.height
        width = self._metadata.width
        tile_size = self.config.tiling.tile_size
        zone_width = self.config.border_correction.zone_width
        current_label = start_label

        borders = BorderTiles(grid, zone_width=zone_width)

        # State directory
        state_dir = output_dir / "state" / "corner"
        state_dir.mkdir(parents=True, exist_ok=True)

        corner_tiles = borders.get_corner_tiles()
        c_cfg = self.config.border_correction.corner_tiles

        # Generate FIXED point grid (same for all corners)
        corner_points = make_corner_grid(
            tile_size=tile_size,
            zone_width=zone_width,
            n_points=c_cfg.grid.n_x  # n_x = n_y for corners
        )
        logger.info(f"Corner grid: {len(corner_points)} points")

        # Output band - write labels directly here
        band2 = np.zeros((height, width), dtype=np.uint32)

        # Pre-generate tile info with filters (no masks list!)
        corner_info = []
        for i, btile in enumerate(corner_tiles):
            corner_info.append({
                'id': f"corner_{i}",
                'btile': btile,
                'filter': lambda m, bx=btile.x, by=btile.y, dx=btile.disc_x, dy=btile.disc_y: crosses_corner(m.mask, dx, dy, bx, by),
            })

        # Process corners pass by pass
        n_passes = c_cfg.cascade.n_passes
        logger.info(f"Processing {len(corner_tiles)} corner tiles")

        for pass_idx in tqdm(range(n_passes), desc="Corner Passes", unit="pass"):
            pass_new = 0
            for info in tqdm(corner_info, desc=f"  Corner Pass {pass_idx}", leave=False, unit="tile"):
                btile = info['btile']
                tile_id = info['id']
                mask_path = state_dir / f"{tile_id}_mask.npy"

                region = load_region(input_path, btile.x, btile.y, btile.width, btile.height)
                combined_mask = load_combined_mask(mask_path, btile.height, btile.width)

                new_masks, combined_mask, coverage = run_single_pass(
                    predictor=self.predictor,
                    image=region,
                    combined_mask=combined_mask,
                    initial_points=corner_points,  # Fixed grid, same for all corners
                    config=c_cfg.cascade,
                    pass_idx=pass_idx,
                    mask_filter=info['filter']
                )

                save_combined_mask(combined_mask, mask_path)

                # Convert masks to labels IMMEDIATELY
                # Only fill empty pixels (don't overwrite existing labels)
                # Skip masks that are too large (>40% of tile)
                tile_pixels = btile.width * btile.height
                max_mask_pixels = int(tile_pixels * 0.4)

                for mask in new_masks:
                    mask_size = mask.mask.sum()
                    if mask_size > max_mask_pixels:
                        continue  # Skip oversized masks

                    y_end = min(btile.y + mask.mask.shape[0], height)
                    x_end = min(btile.x + mask.mask.shape[1], width)
                    local_mask = mask.mask[:y_end - btile.y, :x_end - btile.x]
                    region = band2[btile.y:y_end, btile.x:x_end]
                    region[(local_mask > 0) & (region == 0)] = current_label
                    current_label += 1
                    pass_new += 1

            logger.debug(f"Corner Pass {pass_idx}: {pass_new} new masks")

        if self.config.output.save_intermediate:
            save_labels(
                band2,
                output_dir / "band2_corners.tif",
                crs=self._metadata.crs,
                transform=self._metadata.transform
            )

        return band2, current_label
