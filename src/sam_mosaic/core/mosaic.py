"""Mosaic storage backends for tile-based processing.

This module provides two storage strategies for assembling segmentation results:
- InMemoryMosaic: Keeps everything in RAM (fast, but memory-intensive)
- DiskMosaic: Streams tiles to disk (slower I/O, but memory-efficient)

The choice between them is controlled by the `streaming_mode` config option.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import tempfile
import shutil


class MosaicWriter(ABC):
    """Abstract base class for mosaic storage backends.

    Provides a unified interface for writing tiles and reading the complete
    mosaic, regardless of whether data is stored in RAM or on disk.
    """

    @abstractmethod
    def write_tile(self, labels: np.ndarray, row: int, col: int) -> None:
        """Write a tile to the mosaic at the specified position.

        Args:
            labels: Label array for the tile (tile_size x tile_size).
            row: Tile row index.
            col: Tile column index.
        """
        pass

    @abstractmethod
    def read(self) -> np.ndarray:
        """Read the complete mosaic into memory.

        Returns:
            Complete mosaic array (height x width).
        """
        pass

    @abstractmethod
    def copy(self) -> np.ndarray:
        """Create a copy of the current mosaic state.

        Returns:
            Copy of the mosaic array.
        """
        pass

    @abstractmethod
    def update(self, data: np.ndarray) -> None:
        """Update the mosaic with new data (e.g., after merge operations).

        Args:
            data: New mosaic data to store.
        """
        pass

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int]:
        """Return mosaic dimensions (height, width)."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up resources (close files, delete temp files, etc.)."""
        pass


class InMemoryMosaic(MosaicWriter):
    """RAM-based mosaic storage.

    Stores the entire mosaic in a numpy array. Fast for small to medium
    images, but memory usage grows with image size.

    Memory usage: height * width * 4 bytes (uint32)
    Example: 10k x 10k = 400 MB
    """

    def __init__(self, height: int, width: int, tile_size: int):
        """Initialize in-memory mosaic.

        Args:
            height: Mosaic height in pixels.
            width: Mosaic width in pixels.
            tile_size: Size of each tile in pixels.
        """
        self._data = np.zeros((height, width), dtype=np.uint32)
        self._tile_size = tile_size
        self._height = height
        self._width = width

    def write_tile(self, labels: np.ndarray, row: int, col: int) -> None:
        """Write tile directly to the numpy array."""
        y_start = row * self._tile_size
        x_start = col * self._tile_size
        self._data[y_start:y_start + self._tile_size,
                   x_start:x_start + self._tile_size] = labels

    def read(self) -> np.ndarray:
        """Return reference to the mosaic array."""
        return self._data

    def copy(self) -> np.ndarray:
        """Return a copy of the mosaic array."""
        return self._data.copy()

    def update(self, data: np.ndarray) -> None:
        """Update mosaic with new data."""
        self._data = data

    @property
    def shape(self) -> Tuple[int, int]:
        return (self._height, self._width)

    def close(self) -> None:
        """No cleanup needed for in-memory storage."""
        pass


class DiskMosaic(MosaicWriter):
    """Disk-based mosaic storage using rasterio.

    Streams tiles to a temporary GeoTIFF file, minimizing RAM usage.
    Only loads data into memory when explicitly requested via read().

    Memory usage during processing: ~1 tile at a time
    Disk usage: height * width * 4 bytes (same as RAM version)
    """

    def __init__(
        self,
        height: int,
        width: int,
        tile_size: int,
        crs: Optional[object] = None,
        transform: Optional[object] = None
    ):
        """Initialize disk-based mosaic.

        Args:
            height: Mosaic height in pixels.
            width: Mosaic width in pixels.
            tile_size: Size of each tile in pixels.
            crs: Coordinate reference system for the output.
            transform: Affine transform for georeferencing.
        """
        import rasterio
        from rasterio.transform import Affine

        self._tile_size = tile_size
        self._height = height
        self._width = width
        self._crs = crs
        self._transform = transform or Affine.identity()

        # Create temporary file
        self._temp_dir = tempfile.mkdtemp(prefix="sam_mosaic_")
        self._temp_path = Path(self._temp_dir) / "mosaic_temp.tif"

        # Create the GeoTIFF file
        profile = {
            "driver": "GTiff",
            "dtype": "uint32",
            "width": width,
            "height": height,
            "count": 1,
            "crs": crs,
            "transform": self._transform,
            "tiled": True,
            "blockxsize": min(512, width),
            "blockysize": min(512, height),
            "compress": "lzw",
        }

        # Initialize with zeros
        self._dataset = rasterio.open(self._temp_path, "w+", **profile)

        # Write zeros to initialize (rasterio requires explicit initialization)
        # Do this in chunks to avoid memory spike
        chunk_size = 1024
        zeros = np.zeros((chunk_size, width), dtype=np.uint32)
        for y in range(0, height, chunk_size):
            h = min(chunk_size, height - y)
            if h < chunk_size:
                zeros = np.zeros((h, width), dtype=np.uint32)
            window = rasterio.windows.Window(0, y, width, h)
            self._dataset.write(zeros, 1, window=window)

    def write_tile(self, labels: np.ndarray, row: int, col: int) -> None:
        """Write tile to the GeoTIFF using windowed write."""
        import rasterio

        y_start = row * self._tile_size
        x_start = col * self._tile_size

        window = rasterio.windows.Window(
            x_start, y_start,
            self._tile_size, self._tile_size
        )

        self._dataset.write(labels.astype(np.uint32), 1, window=window)

    def read(self) -> np.ndarray:
        """Read the complete mosaic into memory."""
        return self._dataset.read(1)

    def copy(self) -> np.ndarray:
        """Read and return a copy of the mosaic."""
        return self.read().copy()

    def update(self, data: np.ndarray) -> None:
        """Update the entire mosaic with new data."""
        import rasterio

        # Write in chunks to avoid memory issues
        chunk_size = 1024
        for y in range(0, self._height, chunk_size):
            h = min(chunk_size, self._height - y)
            window = rasterio.windows.Window(0, y, self._width, h)
            self._dataset.write(data[y:y+h, :].astype(np.uint32), 1, window=window)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self._height, self._width)

    def close(self) -> None:
        """Close the dataset and clean up temporary files."""
        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None

        # Clean up temp directory
        if hasattr(self, '_temp_dir') and Path(self._temp_dir).exists():
            shutil.rmtree(self._temp_dir, ignore_errors=True)

    def __del__(self):
        """Ensure cleanup on garbage collection."""
        self.close()


def create_mosaic_writer(
    height: int,
    width: int,
    tile_size: int,
    streaming_mode: str = "auto",
    crs: Optional[object] = None,
    transform: Optional[object] = None
) -> MosaicWriter:
    """Factory function to create the appropriate mosaic writer.

    Args:
        height: Mosaic height in pixels.
        width: Mosaic width in pixels.
        tile_size: Size of each tile in pixels.
        streaming_mode: "auto", "ram", or "disk".
        crs: Coordinate reference system (for disk mode).
        transform: Affine transform (for disk mode).

    Returns:
        MosaicWriter instance (InMemoryMosaic or DiskMosaic).
    """
    if streaming_mode == "ram":
        return InMemoryMosaic(height, width, tile_size)

    elif streaming_mode == "disk":
        return DiskMosaic(height, width, tile_size, crs, transform)

    elif streaming_mode == "auto":
        # Estimate memory usage
        mosaic_bytes = height * width * 4  # uint32

        # Get available RAM
        try:
            import psutil
            available_ram = psutil.virtual_memory().available
            ram_threshold = available_ram * 0.3  # Use disk if mosaic > 30% of available RAM
        except ImportError:
            # If psutil not available, use conservative threshold (1 GB)
            ram_threshold = 1 * 1024 * 1024 * 1024

        if mosaic_bytes > ram_threshold:
            return DiskMosaic(height, width, tile_size, crs, transform)
        else:
            return InMemoryMosaic(height, width, tile_size)

    else:
        raise ValueError(f"Unknown streaming_mode: {streaming_mode}")
