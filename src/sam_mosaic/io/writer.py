"""Output writing utilities."""

from pathlib import Path
from typing import Optional, Union
import numpy as np
import rasterio
from rasterio.transform import Affine


def save_labels(
    labels: np.ndarray,
    path: Union[str, Path],
    crs: Optional[object] = None,
    transform: Optional[Affine] = None,
    nodata: int = 0
) -> None:
    """Save label array as GeoTIFF.

    Args:
        labels: Label array of shape (H, W) with integer labels.
        path: Output file path.
        crs: Coordinate reference system.
        transform: Affine transform for georeferencing.
        nodata: NoData value (default 0 for background).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Determine appropriate dtype
    max_label = labels.max()
    if max_label <= 255:
        dtype = np.uint8
    elif max_label <= 65535:
        dtype = np.uint16
    else:
        dtype = np.uint32

    height, width = labels.shape

    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform or Affine.identity(),
        "nodata": nodata,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(labels.astype(dtype), 1)


def save_mask(
    mask: np.ndarray,
    path: Union[str, Path],
    crs: Optional[object] = None,
    transform: Optional[Affine] = None
) -> None:
    """Save binary mask as GeoTIFF.

    Args:
        mask: Binary mask array of shape (H, W).
        path: Output file path.
        crs: Coordinate reference system.
        transform: Affine transform for georeferencing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    height, width = mask.shape

    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": np.uint8,
        "crs": crs,
        "transform": transform or Affine.identity(),
        "nodata": 0,
        "compress": "lzw",
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mask.astype(np.uint8), 1)


def save_rgb_preview(
    image: np.ndarray,
    labels: np.ndarray,
    path: Union[str, Path],
    alpha: float = 0.5
) -> None:
    """Save RGB preview with colored labels overlay.

    Args:
        image: RGB image (H, W, 3).
        labels: Label array (H, W).
        path: Output file path.
        alpha: Overlay transparency (0-1).
    """
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Create random colors for labels
    n_labels = labels.max() + 1
    colors = np.random.randint(0, 255, (n_labels, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background is black

    # Create colored label image
    colored = colors[labels]

    # Blend with original
    blended = (
        image.astype(np.float32) * (1 - alpha) +
        colored.astype(np.float32) * alpha
    ).astype(np.uint8)

    # Save
    Image.fromarray(blended).save(path)
