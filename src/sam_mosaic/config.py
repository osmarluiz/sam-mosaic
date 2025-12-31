"""Configuration loading and validation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional
import yaml


@dataclass
class ThresholdsConfig:
    """Threshold range for cascade passes (start -> end)."""
    iou: Tuple[float, float] = (0.88, 0.70)
    stability: Tuple[float, float] = (0.92, 0.75)

    def interpolate(self, pass_idx: int, n_passes: int) -> Tuple[float, float]:
        """Get thresholds for a specific pass (linear interpolation)."""
        if n_passes <= 1:
            return self.iou[0], self.stability[0]
        t = pass_idx / (n_passes - 1)
        iou = self.iou[0] + t * (self.iou[1] - self.iou[0])
        stability = self.stability[0] + t * (self.stability[1] - self.stability[0])
        return iou, stability


@dataclass
class CascadeConfig:
    """Cascade refinement configuration."""
    n_passes: int = 20
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    points_per_pass: int = 64           # K-means points for pass 2+
    point_erosion: int = 5              # Erosion pixels for point selection


@dataclass
class UniformGridConfig:
    """Uniform grid configuration for base tiles."""
    points_per_side: int = 64  # 64x64 = 4096 points


@dataclass
class RectGridConfig:
    """Rectangular grid configuration for V/H tiles."""
    n_across: int = 5    # perpendicular to discontinuity
    n_along: int = 100   # parallel to discontinuity


@dataclass
class CornerGridConfig:
    """Grid configuration for corner tiles."""
    n_x: int = 5
    n_y: int = 5


@dataclass
class BaseTilesConfig:
    """Base tile segmentation configuration."""
    grid: UniformGridConfig = field(default_factory=UniformGridConfig)
    cascade: CascadeConfig = field(default_factory=lambda: CascadeConfig(
        n_passes=20,
        thresholds=ThresholdsConfig(
            iou=(0.92, 0.56),
            stability=(0.95, 0.59)
        )
    ))


@dataclass
class VTilesConfig:
    """Vertical discontinuity tiles configuration."""
    grid: RectGridConfig = field(default_factory=RectGridConfig)
    cascade: CascadeConfig = field(default_factory=lambda: CascadeConfig(
        n_passes=5,
        thresholds=ThresholdsConfig(
            iou=(0.88, 0.70),
            stability=(0.92, 0.75)
        )
    ))


@dataclass
class HTilesConfig:
    """Horizontal discontinuity tiles configuration."""
    grid: RectGridConfig = field(default_factory=RectGridConfig)
    cascade: CascadeConfig = field(default_factory=lambda: CascadeConfig(
        n_passes=5,
        thresholds=ThresholdsConfig(
            iou=(0.88, 0.70),
            stability=(0.92, 0.75)
        )
    ))


@dataclass
class CornerTilesConfig:
    """Corner tiles configuration."""
    grid: CornerGridConfig = field(default_factory=CornerGridConfig)
    cascade: CascadeConfig = field(default_factory=lambda: CascadeConfig(
        n_passes=5,
        thresholds=ThresholdsConfig(
            iou=(0.88, 0.70),
            stability=(0.92, 0.75)
        )
    ))


@dataclass
class BorderCorrectionConfig:
    """Border correction configuration."""
    zone_width: int = 100
    v_tiles: VTilesConfig = field(default_factory=VTilesConfig)
    h_tiles: HTilesConfig = field(default_factory=HTilesConfig)
    corner_tiles: CornerTilesConfig = field(default_factory=CornerTilesConfig)


@dataclass
class MergeConfig:
    """Multi-band merge configuration."""
    priority: List[int] = field(default_factory=lambda: [2, 1, 0])  # corners > V/H > base


@dataclass
class OutputConfig:
    """Output configuration."""
    save_intermediate: bool = True
    formats: List[str] = field(default_factory=lambda: ["tif", "shp"])


@dataclass
class SAMConfig:
    """SAM model configuration."""
    checkpoint: str = ""
    model_type: str = "vit_h"
    device: str = "cuda"


@dataclass
class TilingConfig:
    """Tiling configuration."""
    tile_size: int = 1024
    overlap: int = 0


@dataclass
class Config:
    """Main configuration container."""
    sam: SAMConfig = field(default_factory=SAMConfig)
    tiling: TilingConfig = field(default_factory=TilingConfig)
    base_tiles: BaseTilesConfig = field(default_factory=BaseTilesConfig)
    border_correction: BorderCorrectionConfig = field(default_factory=BorderCorrectionConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _parse_thresholds(data: dict) -> ThresholdsConfig:
    """Parse thresholds from dict."""
    return ThresholdsConfig(
        iou=tuple(data.get("iou", [0.88, 0.70])),
        stability=tuple(data.get("stability", [0.92, 0.75]))
    )


def _parse_cascade(data: dict) -> CascadeConfig:
    """Parse cascade config from dict."""
    thresholds = _parse_thresholds(data.get("thresholds", {}))
    return CascadeConfig(
        n_passes=data.get("n_passes", 20),
        thresholds=thresholds,
        points_per_pass=data.get("points_per_pass", 64),
        point_erosion=data.get("point_erosion", 5)
    )


def load_config(path: str | Path) -> Config:
    """Load configuration from YAML file.

    Args:
        path: Path to YAML configuration file.

    Returns:
        Parsed Config object.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    config = Config()

    # SAM config
    if "sam" in data:
        sam_data = data["sam"]
        config.sam = SAMConfig(
            checkpoint=sam_data.get("checkpoint", ""),
            model_type=sam_data.get("model_type", "vit_h"),
            device=sam_data.get("device", "cuda")
        )

    # Tiling config
    if "tiling" in data:
        tile_data = data["tiling"]
        config.tiling = TilingConfig(
            tile_size=tile_data.get("tile_size", 1024),
            overlap=tile_data.get("overlap", 0)
        )

    # Base tiles config
    if "base_tiles" in data:
        bt_data = data["base_tiles"]
        grid_data = bt_data.get("grid", {})
        config.base_tiles = BaseTilesConfig(
            grid=UniformGridConfig(
                points_per_side=grid_data.get("points_per_side", 64)
            ),
            cascade=_parse_cascade(bt_data.get("cascade", {}))
        )

    # Border correction config
    if "border_correction" in data:
        bc_data = data["border_correction"]

        # V tiles
        v_data = bc_data.get("v_tiles", {})
        v_grid = v_data.get("grid", {})
        v_tiles = VTilesConfig(
            grid=RectGridConfig(
                n_across=v_grid.get("n_across", 5),
                n_along=v_grid.get("n_along", 100)
            ),
            cascade=_parse_cascade(v_data.get("cascade", {}))
        )

        # H tiles
        h_data = bc_data.get("h_tiles", {})
        h_grid = h_data.get("grid", {})
        h_tiles = HTilesConfig(
            grid=RectGridConfig(
                n_across=h_grid.get("n_across", 5),
                n_along=h_grid.get("n_along", 100)
            ),
            cascade=_parse_cascade(h_data.get("cascade", {}))
        )

        # Corner tiles
        c_data = bc_data.get("corner_tiles", {})
        c_grid = c_data.get("grid", {})
        corner_tiles = CornerTilesConfig(
            grid=CornerGridConfig(
                n_x=c_grid.get("n_x", 5),
                n_y=c_grid.get("n_y", 5)
            ),
            cascade=_parse_cascade(c_data.get("cascade", {}))
        )

        config.border_correction = BorderCorrectionConfig(
            zone_width=bc_data.get("zone_width", 100),
            v_tiles=v_tiles,
            h_tiles=h_tiles,
            corner_tiles=corner_tiles
        )

    # Merge config
    if "merge" in data:
        merge_data = data["merge"]
        config.merge = MergeConfig(
            priority=merge_data.get("priority", [2, 1, 0])
        )

    # Output config
    if "output" in data:
        out_data = data["output"]
        config.output = OutputConfig(
            save_intermediate=out_data.get("save_intermediate", True),
            formats=out_data.get("formats", ["tif", "shp"])
        )

    return config
