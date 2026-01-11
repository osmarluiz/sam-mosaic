"""Image and tile reading utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import rasterio
from rasterio.windows import Window


@dataclass
class ImageMetadata:
    """Metadata about an input image.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        bands: Number of bands.
        dtype: Data type of pixels.
        crs: Coordinate reference system.
        transform: Affine transform.
        nodata: NoData value if set.
    """
    width: int
    height: int
    bands: int
    dtype: str
    crs: Optional[object] = None
    transform: Optional[object] = None
    nodata: Optional[float] = None


@dataclass
class TileInfo:
    """Information about a loaded tile with padding.

    Attributes:
        data: Tile image data (H, W, C) or (H, W).
        crop_x: X offset to crop useful area from padded tile.
        crop_y: Y offset to crop useful area from padded tile.
        tile_size: Size of useful area (after cropping).
        row: Tile row index.
        col: Tile column index.
    """
    data: np.ndarray
    crop_x: int
    crop_y: int
    tile_size: int
    row: int
    col: int


def get_image_metadata(path: Union[str, Path]) -> ImageMetadata:
    """Get metadata about an image without loading it.

    Args:
        path: Path to image file.

    Returns:
        ImageMetadata object.

    Raises:
        FileNotFoundError: If image file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with rasterio.open(path) as src:
        return ImageMetadata(
            width=src.width,
            height=src.height,
            bands=src.count,
            dtype=str(src.dtypes[0]),
            crs=src.crs,
            transform=src.transform,
            nodata=src.nodata,
        )


def load_image(
    path: Union[str, Path],
    bands: Optional[Tuple[int, ...]] = None
) -> np.ndarray:
    """Load entire image into memory.

    Args:
        path: Path to image file.
        bands: Optional tuple of band indices (1-based). Default loads all.

    Returns:
        Image array of shape (H, W, C) for multi-band or (H, W) for single band.
    """
    with rasterio.open(path) as src:
        if bands is None:
            bands = tuple(range(1, src.count + 1))

        data = src.read(bands)

        # Convert from (C, H, W) to (H, W, C)
        if data.ndim == 3:
            data = np.transpose(data, (1, 2, 0))

        return data


def load_tile(
    path: Union[str, Path],
    row: int,
    col: int,
    tile_size: int,
    padding: int,
    bands: Optional[Tuple[int, ...]] = None
) -> TileInfo:
    """Load a tile with adaptive padding.

    Uses adaptive stride for border tiles to avoid reading outside
    image bounds while maintaining discontinuity positions.

    The crop_x and crop_y values indicate where to extract the
    useful tile_size x tile_size area from the padded tile:
    - First row/col: crop_x/y = 0 (padding is on right/bottom)
    - Middle row/col: crop_x/y = padding (padding on both sides)
    - Last row/col: crop_x/y = 2*padding (padding is on left/top)

    Args:
        path: Path to image file.
        row: Tile row index (0-based).
        col: Tile column index (0-based).
        tile_size: Size of useful tile area.
        padding: Extra pixels to load on each side.
        bands: Optional tuple of band indices (1-based).

    Returns:
        TileInfo with tile data and crop coordinates.
    """
    with rasterio.open(path) as src:
        if bands is None:
            bands = tuple(range(1, src.count + 1))

        img_width = src.width
        img_height = src.height

        # Calculate number of tiles
        n_cols = img_width // tile_size
        n_rows = img_height // tile_size

        # Base tile position
        tile_x = col * tile_size
        tile_y = row * tile_size

        # Padded tile size
        pad_size = tile_size + 2 * padding

        # Adaptive stride: adjust for border tiles
        # X axis
        if col == 0:
            # First column: no left padding
            read_x = 0
            crop_x = 0
        elif col == n_cols - 1:
            # Last column: no right padding
            read_x = img_width - pad_size
            crop_x = 2 * padding
        else:
            # Middle columns: padding on both sides
            read_x = tile_x - padding
            crop_x = padding

        # Y axis
        if row == 0:
            # First row: no top padding
            read_y = 0
            crop_y = 0
        elif row == n_rows - 1:
            # Last row: no bottom padding
            read_y = img_height - pad_size
            crop_y = 2 * padding
        else:
            # Middle rows: padding on both sides
            read_y = tile_y - padding
            crop_y = padding

        # Read the tile
        window = Window(
            col_off=read_x,
            row_off=read_y,
            width=pad_size,
            height=pad_size
        )

        data = src.read(bands, window=window)

        # Convert from (C, H, W) to (H, W, C)
        if data.ndim == 3:
            data = np.transpose(data, (1, 2, 0))

        return TileInfo(
            data=data,
            crop_x=crop_x,
            crop_y=crop_y,
            tile_size=tile_size,
            row=row,
            col=col,
        )


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    """Ensure image is uint8 format for SAM.

    Args:
        image: Input image array.

    Returns:
        Image as uint8.

    Raises:
        ValueError: If conversion not supported.
    """
    if image.dtype == np.uint8:
        return image

    if image.dtype == np.uint16:
        # Check if values are already in 0-255 range
        if image.max() <= 255:
            return image.astype(np.uint8)
        # Otherwise scale from 16-bit
        return (image / 256).astype(np.uint8)

    if image.dtype in [np.float32, np.float64]:
        # Assume 0-1 range
        return (image * 255).clip(0, 255).astype(np.uint8)

    raise ValueError(f"Unsupported image dtype: {image.dtype}")


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Ensure image has 3 channels for SAM.

    Args:
        image: Input image array.

    Returns:
        RGB image (H, W, 3).
    """
    if image.ndim == 2:
        # Single band -> replicate to 3 channels
        return np.stack([image, image, image], axis=2)

    if image.ndim == 3:
        if image.shape[2] == 3:
            return image
        if image.shape[2] == 1:
            # Single band -> replicate
            return np.concatenate([image, image, image], axis=2)
        if image.shape[2] > 3:
            # Take first 3 bands
            return image[:, :, :3]

    raise ValueError(f"Unsupported image shape: {image.shape}")
