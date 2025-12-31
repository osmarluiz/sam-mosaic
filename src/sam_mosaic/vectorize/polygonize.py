"""Vectorization and shapefile export."""

from pathlib import Path
from typing import List, Tuple, Optional, Any, Dict

import numpy as np


def masks_to_polygons(
    labels: np.ndarray,
    transform: Optional[Any] = None,
    crs: Optional[Any] = None,
    simplify_tolerance: float = 1.0,
    min_area: int = 0
) -> List[Dict]:
    """Convert label raster to polygon features.

    Args:
        labels: Label array (H, W) where each unique value is an instance.
        transform: Affine transform for georeferencing.
        crs: Coordinate reference system.
        simplify_tolerance: Polygon simplification tolerance in pixels.
        min_area: Minimum area in pixels (filter small fragments).

    Returns:
        List of feature dictionaries with 'geometry' and 'properties'.
    """
    try:
        from rasterio.features import shapes
        from shapely.geometry import shape, mapping
        from shapely.validation import make_valid
    except ImportError:
        raise ImportError("Requires rasterio and shapely for vectorization")

    # Pre-calculate pixel areas for ALL labels at once - O(n) numpy vectorized
    unique_labels, counts = np.unique(labels, return_counts=True)
    area_dict = dict(zip(unique_labels, counts))

    features = []

    # Generate shapes from raster
    for geom, value in shapes(labels.astype(np.int32), transform=transform):
        if value == 0:  # Skip background
            continue

        # Early rejection of small polygons (before creating shapely objects)
        area_pixels = area_dict.get(int(value), 0)
        if area_pixels < min_area:
            continue

        # Convert to shapely geometry
        poly = shape(geom)

        # Make valid and simplify
        if not poly.is_valid:
            poly = make_valid(poly)
        if simplify_tolerance > 0:
            poly = poly.simplify(simplify_tolerance, preserve_topology=True)

        feature = {
            "geometry": mapping(poly),
            "properties": {
                "id": int(value),
                "area_pixels": area_pixels,
                "area_geo": poly.area
            }
        }
        features.append(feature)

    return features


def save_shapefile(
    features: List[Dict],
    path: str | Path,
    crs: Optional[Any] = None
) -> None:
    """Save features to shapefile.

    Args:
        features: List of feature dictionaries.
        path: Output path (without extension).
        crs: Coordinate reference system.
    """
    try:
        import fiona
        from fiona.crs import from_epsg
    except ImportError:
        raise ImportError("Requires fiona for shapefile export")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure .shp extension
    if path.suffix != ".shp":
        path = path.with_suffix(".shp")

    if not features:
        return

    # Determine geometry type from first feature
    geom_type = features[0]["geometry"]["type"]
    if "Multi" in geom_type:
        geom_type = "MultiPolygon"
    else:
        geom_type = "Polygon"

    # Define schema
    schema = {
        "geometry": geom_type,
        "properties": {
            "id": "int",
            "area_pixels": "int",
            "area_geo": "float"
        }
    }

    with fiona.open(
        str(path),
        "w",
        driver="ESRI Shapefile",
        crs=crs,
        schema=schema
    ) as dst:
        for feat in features:
            dst.write(feat)


def save_geojson(
    features: List[Dict],
    path: str | Path,
    crs: Optional[Any] = None
) -> None:
    """Save features to GeoJSON.

    Args:
        features: List of feature dictionaries.
        path: Output path.
        crs: Coordinate reference system (ignored for GeoJSON, always WGS84).
    """
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix != ".geojson":
        path = path.with_suffix(".geojson")

    geojson = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", **feat}
            for feat in features
        ]
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(geojson, f)


def save_geopackage(
    features: List[Dict],
    path: str | Path,
    layer_name: str = "segments",
    crs: Optional[Any] = None
) -> None:
    """Save features to GeoPackage.

    Args:
        features: List of feature dictionaries.
        path: Output path.
        layer_name: Layer name in the GeoPackage.
        crs: Coordinate reference system.
    """
    try:
        import fiona
    except ImportError:
        raise ImportError("Requires fiona for GeoPackage export")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix != ".gpkg":
        path = path.with_suffix(".gpkg")

    if not features:
        return

    geom_type = features[0]["geometry"]["type"]
    if "Multi" in geom_type:
        geom_type = "MultiPolygon"
    else:
        geom_type = "Polygon"

    schema = {
        "geometry": geom_type,
        "properties": {
            "id": "int",
            "area_pixels": "int",
            "area_geo": "float"
        }
    }

    with fiona.open(
        str(path),
        "w",
        driver="GPKG",
        layer=layer_name,
        crs=crs,
        schema=schema
    ) as dst:
        for feat in features:
            dst.write(feat)
