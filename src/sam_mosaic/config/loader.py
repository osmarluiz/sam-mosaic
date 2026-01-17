"""YAML configuration loading and saving."""

from pathlib import Path
from typing import Union
import yaml

from sam_mosaic.config.schema import (
    Config,
    TileConfig,
    ThresholdConfig,
    SegmentationConfig,
    MergeConfig,
    OutputConfig,
)


def load_config(path: Union[str, Path]) -> Config:
    """Load configuration from YAML file.

    Args:
        path: Path to YAML configuration file.

    Returns:
        Config object with loaded values.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If YAML parsing fails.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    return _dict_to_config(data)


def save_config(config: Config, path: Union[str, Path]) -> None:
    """Save configuration to YAML file.

    Args:
        config: Config object to save.
        path: Output path for YAML file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = _config_to_dict(config)

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def get_default_config() -> Config:
    """Get default configuration.

    Returns:
        Config object with default values.
    """
    return Config()


def _dict_to_config(data: dict) -> Config:
    """Convert dictionary to Config object.

    Args:
        data: Dictionary from YAML.

    Returns:
        Config object.
    """
    tile_data = data.get("tile", {})
    threshold_data = data.get("threshold", {})
    segmentation_data = data.get("segmentation", {})
    merge_data = data.get("merge", {})
    output_data = data.get("output", {})

    return Config(
        tile=TileConfig(
            size=tile_data.get("size", 1000),
            padding=tile_data.get("padding", 50),
        ),
        threshold=ThresholdConfig(
            iou_start=threshold_data.get("iou_start", 0.93),
            iou_end=threshold_data.get("iou_end", 0.60),
            stability_start=threshold_data.get("stability_start", 0.93),
            stability_end=threshold_data.get("stability_end", 0.60),
            step=threshold_data.get("step", 0.01),
        ),
        segmentation=SegmentationConfig(
            points_per_side=segmentation_data.get("points_per_side", 64),
            target_coverage=segmentation_data.get("target_coverage", 99.0),
            max_passes=segmentation_data.get("max_passes"),
            use_black_mask=segmentation_data.get("use_black_mask", True),
            use_adaptive_threshold=segmentation_data.get("use_adaptive_threshold", True),
            point_strategy=segmentation_data.get("point_strategy", "kmeans"),
            erosion_iterations=segmentation_data.get("erosion_iterations", 10),
        ),
        merge=MergeConfig(
            min_contact_pixels=merge_data.get("min_contact_pixels", 5),
            min_mask_area=merge_data.get("min_mask_area", 100),
            merge_enclosed_max_area=merge_data.get("merge_enclosed_max_area", 500),
        ),
        output=OutputConfig(
            save_labels=output_data.get("save_labels", True),
            save_shapefile=output_data.get("save_shapefile", True),
            save_geopackage=output_data.get("save_geopackage", False),
            save_stats=output_data.get("save_stats", True),
            simplify_tolerance=output_data.get("simplify_tolerance", 1.0),
        ),
        sam_checkpoint=data.get("sam_checkpoint"),
    )


def _config_to_dict(config: Config) -> dict:
    """Convert Config object to dictionary.

    Args:
        config: Config object.

    Returns:
        Dictionary suitable for YAML.
    """
    return {
        "tile": {
            "size": config.tile.size,
            "padding": config.tile.padding,
        },
        "threshold": {
            "iou_start": config.threshold.iou_start,
            "iou_end": config.threshold.iou_end,
            "stability_start": config.threshold.stability_start,
            "stability_end": config.threshold.stability_end,
            "step": config.threshold.step,
        },
        "segmentation": {
            "points_per_side": config.segmentation.points_per_side,
            "target_coverage": config.segmentation.target_coverage,
            "max_passes": config.segmentation.max_passes,
            "use_black_mask": config.segmentation.use_black_mask,
            "use_adaptive_threshold": config.segmentation.use_adaptive_threshold,
            "point_strategy": config.segmentation.point_strategy,
            "erosion_iterations": config.segmentation.erosion_iterations,
        },
        "merge": {
            "min_contact_pixels": config.merge.min_contact_pixels,
            "min_mask_area": config.merge.min_mask_area,
            "merge_enclosed_max_area": config.merge.merge_enclosed_max_area,
        },
        "output": {
            "save_labels": config.output.save_labels,
            "save_shapefile": config.output.save_shapefile,
            "save_geopackage": config.output.save_geopackage,
            "save_stats": config.output.save_stats,
            "simplify_tolerance": config.output.simplify_tolerance,
        },
        "sam_checkpoint": config.sam_checkpoint,
    }
