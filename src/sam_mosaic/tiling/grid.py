"""Tile grid management."""

from dataclasses import dataclass
from typing import Iterator, Tuple, List


@dataclass
class Tile:
    """A single tile in the grid."""
    row: int
    col: int
    x: int      # left edge
    y: int      # top edge
    width: int
    height: int

    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        """Return (x_min, y_min, x_max, y_max)."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    @property
    def center(self) -> Tuple[int, int]:
        """Return (x, y) of tile center."""
        return (self.x + self.width // 2, self.y + self.height // 2)


class TileGrid:
    """Grid of tiles covering an image."""

    def __init__(
        self,
        image_width: int,
        image_height: int,
        tile_size: int,
        overlap: int = 0
    ):
        """Initialize tile grid.

        Args:
            image_width: Full image width in pixels.
            image_height: Full image height in pixels.
            tile_size: Size of each tile (square).
            overlap: Overlap between adjacent tiles.
        """
        self.image_width = image_width
        self.image_height = image_height
        self.tile_size = tile_size
        self.overlap = overlap

        # Calculate step and grid dimensions
        self.step = tile_size - overlap
        self.n_cols = max(1, (image_width - overlap) // self.step)
        self.n_rows = max(1, (image_height - overlap) // self.step)

        # Adjust if image doesn't divide evenly
        if self.n_cols * self.step + overlap < image_width:
            self.n_cols += 1
        if self.n_rows * self.step + overlap < image_height:
            self.n_rows += 1

    def __len__(self) -> int:
        """Total number of tiles."""
        return self.n_rows * self.n_cols

    def __iter__(self) -> Iterator[Tile]:
        """Iterate over all tiles."""
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                yield self.get_tile(row, col)

    def get_tile(self, row: int, col: int) -> Tile:
        """Get tile at specific grid position.

        Args:
            row: Row index (0-based).
            col: Column index (0-based).

        Returns:
            Tile object with position and size.
        """
        x = col * self.step
        y = row * self.step

        # Clamp to image bounds
        width = min(self.tile_size, self.image_width - x)
        height = min(self.tile_size, self.image_height - y)

        return Tile(row=row, col=col, x=x, y=y, width=width, height=height)

    def tiles_in_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int
    ) -> List[Tile]:
        """Get all tiles that intersect a region.

        Args:
            x: Region left edge.
            y: Region top edge.
            width: Region width.
            height: Region height.

        Returns:
            List of tiles intersecting the region.
        """
        x_max = x + width
        y_max = y + height

        tiles = []
        for tile in self:
            tx_min, ty_min, tx_max, ty_max = tile.bounds
            # Check intersection
            if tx_max > x and tx_min < x_max and ty_max > y and ty_min < y_max:
                tiles.append(tile)

        return tiles

    @property
    def v_discontinuities(self) -> List[int]:
        """Get x-coordinates of vertical discontinuity lines.

        These are the boundaries between adjacent column tiles.
        """
        return [col * self.step + self.tile_size for col in range(self.n_cols - 1)]

    @property
    def h_discontinuities(self) -> List[int]:
        """Get y-coordinates of horizontal discontinuity lines.

        These are the boundaries between adjacent row tiles.
        """
        return [row * self.step + self.tile_size for row in range(self.n_rows - 1)]

    @property
    def corner_positions(self) -> List[Tuple[int, int]]:
        """Get (x, y) positions of all tile corners (intersections).

        Returns coordinates where 4 tiles meet.
        """
        corners = []
        for x in self.v_discontinuities:
            for y in self.h_discontinuities:
                corners.append((x, y))
        return corners
