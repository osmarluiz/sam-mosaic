"""SAM-Mosaic: Segment large images using SAM with tile-based processing."""

__version__ = "0.1.0"

from sam_mosaic.config import load_config, Config
from sam_mosaic.pipeline import Pipeline

__all__ = ["load_config", "Config", "Pipeline", "__version__"]
