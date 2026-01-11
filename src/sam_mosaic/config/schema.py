"""Configuration dataclasses for SAM-Mosaic."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TileConfig:
    """Tile processing configuration.

    Attributes:
        size: Tile size in pixels (useful area after cropping).
        padding: Extra pixels on each side for SAM context.
                 Total read size = size + 2 * padding.
    """
    size: int = 1000
    padding: int = 50

    @property
    def padded_size(self) -> int:
        """Total tile size including padding."""
        return self.size + 2 * self.padding


@dataclass
class ThresholdConfig:
    """Threshold configuration for SAM predictions.

    The thresholds decrease from start to end across passes,
    allowing more permissive segmentation in later passes.

    Attributes:
        iou_start: Initial IoU threshold (restrictive).
        iou_end: Final IoU threshold (permissive).
        stability_start: Initial stability threshold.
        stability_end: Final stability threshold.
        step: Threshold decrease per pass.
    """
    iou_start: float = 0.93
    iou_end: float = 0.60
    stability_start: float = 0.93
    stability_end: float = 0.60
    step: float = 0.01

    def get_thresholds_for_pass(self, pass_idx: int) -> tuple[float, float]:
        """Get IoU and stability thresholds for a given pass.

        Args:
            pass_idx: Zero-based pass index.

        Returns:
            Tuple of (iou_threshold, stability_threshold).
        """
        iou = max(self.iou_end, self.iou_start - pass_idx * self.step)
        stability = max(self.stability_end, self.stability_start - pass_idx * self.step)
        return iou, stability


@dataclass
class SegmentationConfig:
    """Segmentation behavior configuration.

    Attributes:
        points_per_side: Grid density for initial point placement.
        target_coverage: Stop segmentation when this coverage % is reached.
        max_passes: Maximum number of passes (None = unlimited).
        use_black_mask: Whether to mask segmented areas with black.
        use_adaptive_threshold: Whether to decrease threshold each pass.
    """
    points_per_side: int = 64
    target_coverage: float = 99.0
    max_passes: Optional[int] = None
    use_black_mask: bool = True
    use_adaptive_threshold: bool = True


@dataclass
class MergeConfig:
    """Post-processing merge configuration.

    Attributes:
        min_contact_pixels: Minimum pixels of contact to merge at discontinuities.
        min_mask_area: Remove masks smaller than this area.
        merge_enclosed_max_area: Merge enclosed masks up to this size.
    """
    min_contact_pixels: int = 5
    min_mask_area: int = 100
    merge_enclosed_max_area: int = 500


@dataclass
class OutputConfig:
    """Output file configuration.

    Attributes:
        save_labels: Whether to save label raster (TIFF).
        save_shapefile: Whether to save vectorized shapefile.
        save_geopackage: Whether to save GeoPackage.
        save_stats: Whether to save detailed stats JSON.
        simplify_tolerance: Polygon simplification tolerance in map units (0 = no simplification).
    """
    save_labels: bool = True
    save_shapefile: bool = True
    save_geopackage: bool = False
    save_stats: bool = True
    simplify_tolerance: float = 1.0


@dataclass
class Config:
    """Main configuration container.

    Attributes:
        tile: Tile processing settings.
        threshold: SAM threshold settings.
        segmentation: Segmentation behavior settings.
        merge: Post-processing merge settings.
        output: Output file settings.
        sam_checkpoint: Path to SAM2 checkpoint file.
    """
    tile: TileConfig = field(default_factory=TileConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    sam_checkpoint: Optional[str] = None

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        if self.tile.size <= 0:
            raise ValueError(f"tile.size must be positive, got {self.tile.size}")
        if self.tile.padding < 0:
            raise ValueError(f"tile.padding must be non-negative, got {self.tile.padding}")
        if not 0 < self.threshold.iou_start <= 1:
            raise ValueError(f"threshold.iou_start must be in (0, 1], got {self.threshold.iou_start}")
        if not 0 < self.threshold.iou_end <= 1:
            raise ValueError(f"threshold.iou_end must be in (0, 1], got {self.threshold.iou_end}")
        if self.threshold.iou_end > self.threshold.iou_start:
            raise ValueError("threshold.iou_end must be <= threshold.iou_start")
        if not 0 < self.segmentation.target_coverage <= 100:
            raise ValueError(f"segmentation.target_coverage must be in (0, 100], got {self.segmentation.target_coverage}")
        if self.segmentation.points_per_side <= 0:
            raise ValueError(f"segmentation.points_per_side must be positive, got {self.segmentation.points_per_side}")

    @classmethod
    def with_overrides(cls, base: "Config", **overrides) -> "Config":
        """Create a new config with overrides applied.

        Args:
            base: Base configuration to start from.
            **overrides: Parameter overrides in flat format (e.g., tile_size=500).

        Returns:
            New Config with overrides applied.
        """
        import copy
        config = copy.deepcopy(base)

        # Map flat parameter names to nested config attributes
        param_map = {
            "tile_size": ("tile", "size"),
            "padding": ("tile", "padding"),
            "iou_start": ("threshold", "iou_start"),
            "iou_end": ("threshold", "iou_end"),
            "stability_start": ("threshold", "stability_start"),
            "stability_end": ("threshold", "stability_end"),
            "threshold_step": ("threshold", "step"),
            "points_per_side": ("segmentation", "points_per_side"),
            "target_coverage": ("segmentation", "target_coverage"),
            "max_passes": ("segmentation", "max_passes"),
            "use_black_mask": ("segmentation", "use_black_mask"),
            "use_adaptive_threshold": ("segmentation", "use_adaptive_threshold"),
            "min_contact_pixels": ("merge", "min_contact_pixels"),
            "min_mask_area": ("merge", "min_mask_area"),
            "merge_enclosed_max_area": ("merge", "merge_enclosed_max_area"),
            "save_labels": ("output", "save_labels"),
            "save_shapefile": ("output", "save_shapefile"),
            "save_geopackage": ("output", "save_geopackage"),
            "simplify_tolerance": ("output", "simplify_tolerance"),
            "sam_checkpoint": (None, "sam_checkpoint"),
        }

        for key, value in overrides.items():
            if value is None:
                continue
            if key not in param_map:
                raise ValueError(f"Unknown parameter: {key}")

            section, attr = param_map[key]
            if section is None:
                setattr(config, attr, value)
            else:
                section_obj = getattr(config, section)
                setattr(section_obj, attr, value)

        return config
