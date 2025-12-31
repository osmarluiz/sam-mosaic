"""Raster output utilities."""

from pathlib import Path
from typing import Optional, Any, Dict

import numpy as np
import rasterio
from rasterio.transform import Affine


def save_raster(
    data: np.ndarray,
    path: str | Path,
    crs: Optional[Any] = None,
    transform: Optional[Affine] = None,
    nodata: Optional[float] = None,
    dtype: Optional[str] = None,
    compress: str = "lzw"
) -> None:
    """Save numpy array as GeoTIFF.

    Args:
        data: Array with shape (H, W) or (H, W, C).
        path: Output path.
        crs: Coordinate reference system.
        transform: Affine transform.
        nodata: NoData value.
        dtype: Output data type. Inferred from data if not specified.
        compress: Compression method.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Determine shape and bands
    if data.ndim == 2:
        height, width = data.shape
        count = 1
        data = data[np.newaxis, :, :]  # Add band dimension
    elif data.ndim == 3:
        height, width, count = data.shape
        data = np.transpose(data, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    else:
        raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")

    # Infer dtype
    if dtype is None:
        dtype = str(data.dtype)

    # Default transform if not provided
    if transform is None:
        transform = Affine.identity()

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress=compress
    ) as dst:
        dst.write(data)


def save_mask(
    mask: np.ndarray,
    path: str | Path,
    crs: Optional[Any] = None,
    transform: Optional[Affine] = None
) -> None:
    """Save binary mask as GeoTIFF.

    Args:
        mask: Boolean or uint8 mask array (H, W).
        path: Output path.
        crs: Coordinate reference system.
        transform: Affine transform.
    """
    # Ensure uint8
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255
    elif mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    save_raster(
        mask,
        path,
        crs=crs,
        transform=transform,
        nodata=0,
        dtype="uint8"
    )


def save_labels(
    labels: np.ndarray,
    path: str | Path,
    crs: Optional[Any] = None,
    transform: Optional[Affine] = None
) -> None:
    """Save label raster (instance segmentation).

    Args:
        labels: Integer label array (H, W) where 0 = background.
        path: Output path.
        crs: Coordinate reference system.
        transform: Affine transform.
    """
    # Choose dtype based on max label
    max_label = labels.max()
    if max_label <= 255:
        dtype = "uint8"
    elif max_label <= 65535:
        dtype = "uint16"
    else:
        dtype = "uint32"

    save_raster(
        labels.astype(dtype),
        path,
        crs=crs,
        transform=transform,
        nodata=0,
        dtype=dtype
    )
