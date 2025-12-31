"""Border detection and discontinuity tile generation."""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from sam_mosaic.tiling.grid import TileGrid


@dataclass
class BorderTile:
    """A tile centered on a discontinuity for border correction."""
    x: int          # left edge
    y: int          # top edge
    width: int
    height: int
    disc_type: str  # "v", "h", or "corner"
    disc_x: int     # x of discontinuity line (for v tiles)
    disc_y: int     # y of discontinuity line (for h tiles)

    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        """Return (x_min, y_min, x_max, y_max)."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)


def find_discontinuities(
    grid: TileGrid
) -> Tuple[List[int], List[int], List[Tuple[int, int]]]:
    """Find all discontinuity locations in a tile grid.

    Args:
        grid: TileGrid instance.

    Returns:
        Tuple of (v_lines, h_lines, corners) where:
        - v_lines: x-coordinates of vertical discontinuities
        - h_lines: y-coordinates of horizontal discontinuities
        - corners: (x, y) positions of corner intersections
    """
    v_lines = grid.v_discontinuities
    h_lines = grid.h_discontinuities
    corners = grid.corner_positions
    return v_lines, h_lines, corners


class BorderTiles:
    """Generate border correction tiles centered on discontinuities.

    Uses same tile size as base tiles (e.g., 2000x2000) for consistency.
    Tiles are positioned so discontinuities fall at the center.
    """

    def __init__(
        self,
        grid: TileGrid,
        zone_width: int = 100
    ):
        """Initialize border tile generator.

        Args:
            grid: TileGrid defining the base tiling.
            zone_width: Width of point zone on each side of discontinuity.
        """
        self.grid = grid
        self.zone_width = zone_width
        self.tile_size = grid.tile_size  # Use same tile size as base
        self.image_width = grid.image_width
        self.image_height = grid.image_height

    def get_v_tiles(self) -> List[BorderTile]:
        """Get tiles for vertical discontinuities.

        Creates 2000x2000 tiles centered on each vertical line.
        Multiple tiles per discontinuity to cover the full height.
        Discontinuity is always at local x = tile_size/2.
        """
        tiles = []
        half = self.tile_size // 2  # e.g., 1000 for 2000px tiles

        for x_line in self.grid.v_discontinuities:
            # Tile x position: center on discontinuity
            tile_x = x_line - half
            tile_x = max(0, tile_x)
            tile_x = min(tile_x, self.image_width - self.tile_size)

            # Create tiles along the full height
            for tile_y in range(0, self.image_height, self.tile_size):
                height = min(self.tile_size, self.image_height - tile_y)
                width = min(self.tile_size, self.image_width - tile_x)

                tile = BorderTile(
                    x=tile_x,
                    y=tile_y,
                    width=width,
                    height=height,
                    disc_type="v",
                    disc_x=x_line,
                    disc_y=0
                )
                tiles.append(tile)
        return tiles

    def get_h_tiles(self) -> List[BorderTile]:
        """Get tiles for horizontal discontinuities.

        Creates 2000x2000 tiles centered on each horizontal line.
        Multiple tiles per discontinuity to cover the full width.
        Discontinuity is always at local y = tile_size/2.
        """
        tiles = []
        half = self.tile_size // 2

        for y_line in self.grid.h_discontinuities:
            # Tile y position: center on discontinuity
            tile_y = y_line - half
            tile_y = max(0, tile_y)
            tile_y = min(tile_y, self.image_height - self.tile_size)

            # Create tiles along the full width
            for tile_x in range(0, self.image_width, self.tile_size):
                width = min(self.tile_size, self.image_width - tile_x)
                height = min(self.tile_size, self.image_height - tile_y)

                tile = BorderTile(
                    x=tile_x,
                    y=tile_y,
                    width=width,
                    height=height,
                    disc_type="h",
                    disc_x=0,
                    disc_y=y_line
                )
                tiles.append(tile)
        return tiles

    def get_corner_tiles(self) -> List[BorderTile]:
        """Get tiles for corner intersections.

        Creates 2000x2000 tiles centered on each corner point.
        Corner is always at local (tile_size/2, tile_size/2).
        """
        tiles = []
        half = self.tile_size // 2

        for x_line, y_line in self.grid.corner_positions:
            # Center tile on corner
            tile_x = x_line - half
            tile_y = y_line - half
            tile_x = max(0, tile_x)
            tile_y = max(0, tile_y)
            tile_x = min(tile_x, self.image_width - self.tile_size)
            tile_y = min(tile_y, self.image_height - self.tile_size)

            width = min(self.tile_size, self.image_width - tile_x)
            height = min(self.tile_size, self.image_height - tile_y)

            tile = BorderTile(
                x=tile_x,
                y=tile_y,
                width=width,
                height=height,
                disc_type="corner",
                disc_x=x_line,
                disc_y=y_line
            )
            tiles.append(tile)
        return tiles


def crosses_v(mask: np.ndarray, disc_x: int, tile_x: int) -> bool:
    """Check if mask crosses a vertical discontinuity line.

    Args:
        mask: Binary mask array (H, W).
        disc_x: Global x-coordinate of the discontinuity.
        tile_x: Global x-coordinate of the tile's left edge.

    Returns:
        True if mask has pixels on both sides of the line.
    """
    local_x = disc_x - tile_x
    if local_x < 0 or local_x >= mask.shape[1]:
        return False

    # Check for pixels on both sides
    left = mask[:, :local_x].any() if local_x > 0 else False
    right = mask[:, local_x:].any() if local_x < mask.shape[1] else False
    return left and right


def crosses_h(mask: np.ndarray, disc_y: int, tile_y: int) -> bool:
    """Check if mask crosses a horizontal discontinuity line.

    Args:
        mask: Binary mask array (H, W).
        disc_y: Global y-coordinate of the discontinuity.
        tile_y: Global y-coordinate of the tile's top edge.

    Returns:
        True if mask has pixels on both sides of the line.
    """
    local_y = disc_y - tile_y
    if local_y < 0 or local_y >= mask.shape[0]:
        return False

    # Check for pixels on both sides
    top = mask[:local_y, :].any() if local_y > 0 else False
    bottom = mask[local_y:, :].any() if local_y < mask.shape[0] else False
    return top and bottom


def crosses_corner(
    mask: np.ndarray,
    disc_x: int,
    disc_y: int,
    tile_x: int,
    tile_y: int
) -> bool:
    """Check if mask crosses a corner (crosses BOTH V and H lines).

    Args:
        mask: Binary mask array (H, W).
        disc_x: Global x-coordinate of the corner.
        disc_y: Global y-coordinate of the corner.
        tile_x: Global x-coordinate of the tile's left edge.
        tile_y: Global y-coordinate of the tile's top edge.

    Returns:
        True if mask crosses both the vertical and horizontal lines.
    """
    local_x = disc_x - tile_x
    local_y = disc_y - tile_y

    if local_x < 0 or local_x >= mask.shape[1]:
        return False
    if local_y < 0 or local_y >= mask.shape[0]:
        return False

    # Check each quadrant
    top_left = mask[:local_y, :local_x].any() if (local_y > 0 and local_x > 0) else False
    top_right = mask[:local_y, local_x:].any() if (local_y > 0 and local_x < mask.shape[1]) else False
    bottom_left = mask[local_y:, :local_x].any() if (local_y < mask.shape[0] and local_x > 0) else False
    bottom_right = mask[local_y:, local_x:].any() if (local_y < mask.shape[0] and local_x < mask.shape[1]) else False

    # Must cross BOTH lines:
    # - Crosses V: has pixels on left AND right of vertical line
    # - Crosses H: has pixels above AND below horizontal line
    crosses_v = (top_left or bottom_left) and (top_right or bottom_right)
    crosses_h = (top_left or top_right) and (bottom_left or bottom_right)

    return crosses_v and crosses_h
