"""Raster to vector conversion (polygonization)."""

from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import numpy as np
import rasterio.features
from rasterio.transform import Affine
from shapely.geometry import shape, mapping


def vectorize_labels(
    labels: np.ndarray,
    output_path: Union[str, Path],
    crs: Optional[object] = None,
    transform: Optional[Affine] = None,
    simplify_tolerance: float = 1.0
) -> int:
    """Convert label raster to vector polygons and save.

    Automatically chooses format based on file extension:
    - .shp -> Shapefile
    - .gpkg -> GeoPackage
    - .geojson -> GeoJSON

    Args:
        labels: Label array of shape (H, W).
        output_path: Output file path.
        crs: Coordinate reference system.
        transform: Affine transform for georeferencing.
        simplify_tolerance: Polygon simplification tolerance (0 = no simplification).

    Returns:
        Number of polygons created.
    """
    output_path = Path(output_path)
    ext = output_path.suffix.lower()

    # Extract polygons
    features = extract_polygons(labels, transform, simplify_tolerance)

    if len(features) == 0:
        return 0

    # Save based on extension
    if ext == ".shp":
        save_shapefile(features, output_path, crs)
    elif ext == ".gpkg":
        save_geopackage(features, output_path, crs)
    elif ext == ".geojson":
        save_geojson(features, output_path, crs)
    else:
        raise ValueError(f"Unsupported output format: {ext}")

    return len(features)


def extract_polygons(
    labels: np.ndarray,
    transform: Optional[Affine] = None,
    simplify_tolerance: float = 1.0
) -> List[Dict[str, Any]]:
    """Extract polygons from label array.

    Uses rasterio.features.shapes for efficient vectorization.

    Args:
        labels: Label array of shape (H, W).
        transform: Affine transform for georeferencing.
        simplify_tolerance: Polygon simplification tolerance (default 1.0).

    Returns:
        List of feature dictionaries with geometry and properties.
    """
    from shapely.validation import make_valid

    if transform is None:
        transform = Affine.identity()

    features = []

    # Use rasterio's efficient shapes function
    for geom, value in rasterio.features.shapes(
        labels.astype(np.int32),
        transform=transform
    ):
        if value == 0:  # Skip background
            continue

        # Convert to shapely for processing
        polygon = shape(geom)

        # Make valid and simplify (matches original)
        if not polygon.is_valid:
            polygon = make_valid(polygon)
        if simplify_tolerance > 0:
            polygon = polygon.simplify(simplify_tolerance, preserve_topology=True)

        features.append({
            "geometry": mapping(polygon),
            "properties": {
                "label_id": int(value),
                "area_m2": polygon.area,
                "perimeter_m": polygon.length,
            }
        })

    return features


def save_shapefile(
    features: List[Dict[str, Any]],
    path: Union[str, Path],
    crs: Optional[object] = None
) -> None:
    """Save features to Shapefile.

    Args:
        features: List of feature dictionaries.
        path: Output file path.
        crs: Coordinate reference system.
    """
    import fiona
    from fiona.crs import from_epsg

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(features) == 0:
        return

    # Define schema
    schema = {
        "geometry": "Polygon",
        "properties": {
            "label_id": "int",
            "area_m2": "float",
            "perimeter_m": "float",
        }
    }

    # Get CRS
    if crs is None:
        crs_dict = from_epsg(4326)  # Default to WGS84
    elif hasattr(crs, "to_dict"):
        crs_dict = crs.to_dict()
    else:
        crs_dict = crs

    with fiona.open(
        str(path),
        "w",
        driver="ESRI Shapefile",
        crs=crs_dict,
        schema=schema
    ) as dst:
        for feature in features:
            dst.write({
                "geometry": feature["geometry"],
                "properties": feature["properties"],
            })


def save_geopackage(
    features: List[Dict[str, Any]],
    path: Union[str, Path],
    crs: Optional[object] = None
) -> None:
    """Save features to GeoPackage.

    Args:
        features: List of feature dictionaries.
        path: Output file path.
        crs: Coordinate reference system.
    """
    import fiona
    from fiona.crs import from_epsg

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(features) == 0:
        return

    schema = {
        "geometry": "Polygon",
        "properties": {
            "label_id": "int",
            "area_m2": "float",
            "perimeter_m": "float",
        }
    }

    if crs is None:
        crs_dict = from_epsg(4326)
    elif hasattr(crs, "to_dict"):
        crs_dict = crs.to_dict()
    else:
        crs_dict = crs

    with fiona.open(
        str(path),
        "w",
        driver="GPKG",
        crs=crs_dict,
        schema=schema,
        layer="segments"
    ) as dst:
        for feature in features:
            dst.write({
                "geometry": feature["geometry"],
                "properties": feature["properties"],
            })


def save_geojson(
    features: List[Dict[str, Any]],
    path: Union[str, Path],
    crs: Optional[object] = None
) -> None:
    """Save features to GeoJSON.

    Args:
        features: List of feature dictionaries.
        path: Output file path.
        crs: Coordinate reference system (ignored for GeoJSON, always WGS84).
    """
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": f["geometry"],
                "properties": f["properties"],
            }
            for f in features
        ]
    }

    with open(path, "w") as f:
        json.dump(geojson, f)
