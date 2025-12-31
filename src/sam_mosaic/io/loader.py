"""Image loading utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import rasterio
from rasterio.windows import Window


@dataclass
class ImageMetadata:
    """Raster image metadata."""
    width: int
    height: int
    bands: int
    dtype: str
    crs: Optional[Any]
    transform: Optional[Any]
    nodata: Optional[float]


def get_metadata(path: str | Path) -> ImageMetadata:
    """Get image metadata without loading full image.

    Args:
        path: Path to raster image.

    Returns:
        ImageMetadata with image properties.
    """
    path = Path(path)
    with rasterio.open(path) as src:
        return ImageMetadata(
            width=src.width,
            height=src.height,
            bands=src.count,
            dtype=str(src.dtypes[0]),
            crs=src.crs,
            transform=src.transform,
            nodata=src.nodata
        )


def load_image(
    path: str | Path,
    window: Optional[Window] = None,
    bands: Optional[Tuple[int, ...]] = None
) -> np.ndarray:
    """Load raster image or a window from it.

    Args:
        path: Path to raster image.
        window: Optional rasterio Window for partial read.
        bands: Optional tuple of band indices (1-based). Default reads all.

    Returns:
        Image array with shape (H, W, C) for multi-band or (H, W) for single band.
    """
    path = Path(path)
    with rasterio.open(path) as src:
        if bands is None:
            bands = tuple(range(1, src.count + 1))

        data = src.read(bands, window=window)

        # Convert from (C, H, W) to (H, W, C) for compatibility with SAM
        if data.ndim == 3:
            data = np.transpose(data, (1, 2, 0))

        return data


def load_tile(
    path: str | Path,
    row: int,
    col: int,
    tile_size: int,
    bands: Optional[Tuple[int, ...]] = None
) -> np.ndarray:
    """Load a specific tile from image.

    Args:
        path: Path to raster image.
        row: Tile row index (0-based).
        col: Tile column index (0-based).
        tile_size: Size of tile in pixels.
        bands: Optional tuple of band indices (1-based).

    Returns:
        Tile array with shape (H, W, C) or (H, W).
    """
    y_off = row * tile_size
    x_off = col * tile_size

    window = Window(
        col_off=x_off,
        row_off=y_off,
        width=tile_size,
        height=tile_size
    )

    return load_image(path, window=window, bands=bands)


def load_region(
    path: str | Path,
    x: int,
    y: int,
    width: int,
    height: int,
    bands: Optional[Tuple[int, ...]] = None
) -> np.ndarray:
    """Load arbitrary rectangular region from image.

    Args:
        path: Path to raster image.
        x: Left edge x-coordinate.
        y: Top edge y-coordinate.
        width: Region width in pixels.
        height: Region height in pixels.
        bands: Optional tuple of band indices (1-based).

    Returns:
        Region array with shape (H, W, C) or (H, W).
    """
    window = Window(col_off=x, row_off=y, width=width, height=height)
    return load_image(path, window=window, bands=bands)
