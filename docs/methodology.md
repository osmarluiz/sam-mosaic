# Methodology

This document describes the SAM-Mosaic methodology in detail.

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Pipeline Overview](#pipeline-overview)
3. [Stage 1: Base Tiles](#stage-1-base-tiles)
4. [Stage 2: Cascade Refinement](#stage-2-cascade-refinement)
5. [Stage 3: Border Correction](#stage-3-border-correction)
6. [Stage 4: Band Merge](#stage-4-band-merge)
7. [Stage 5: Vectorization](#stage-5-vectorization)

---

## Problem Statement

### The Challenge

SAM (Segment Anything Model) produces excellent segmentation results but has memory limitations. A typical GPU cannot process images larger than ~4000×4000 pixels in a single pass.

For large images (e.g., satellite imagery at 15000×30000 pixels), we need tile-based processing. However, this introduces **boundary artifacts**:

```
    Tile A    │    Tile B
              │
   ┌─────┐    │
   │  1  │    │    ┌─────┐
   └─────┘    │    │  2  │
              │    └─────┘
      ┌───────┼───────┐
      │   3a  │  3b   │   ← Object 3 split into fragments!
      └───────┼───────┘
              │
```

Objects crossing tile boundaries are fragmented, reducing segmentation quality.

### Our Solution

We introduce a **3-band merge strategy** that re-segments boundary regions with overlapping tiles, then merges results with priority:

```
Band 0: Base tiles (has fragments at borders)
Band 1: V/H tiles (centered on boundaries, captures crossing objects)
Band 2: Corner tiles (centered on V+H intersections, highest priority)

Final = merge(Band0, Band1, Band2) with priority Band2 > Band1 > Band0
```

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                           INPUT                                      │
│  Large image (PNG, JPEG, TIFF, GeoTIFF, ENVI)                       │
│  + geospatial metadata (CRS, transform)                             │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 1: BASE TILES                              │
│                                                                     │
│  • Divide image into tiles (e.g., 2000×2000)                       │
│  • Run SAM on each tile with dense point grid (64×64)              │
│  • Result: Band 0                                                   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 STAGE 2: CASCADE REFINEMENT                         │
│                                                                     │
│  • Multiple passes with decreasing IoU/stability thresholds        │
│  • Black masking between passes (already segmented → black)        │
│  • Same point grid, different thresholds                           │
│  • Result: Band 0 refined                                          │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 STAGE 3: BORDER CORRECTION                          │
│                                                                     │
│  3A. Detect discontinuities (V and H lines)                        │
│  3B. V/H tiles: centered on each discontinuity → Band 1            │
│  3C. Corner tiles: centered on V+H intersections → Band 2          │
│  Each with its own cascade (starts from zero, independent)         │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 4: BAND MERGE                              │
│                                                                     │
│  Pixel-level NMS with priority:                                    │
│  result = Band0                                                     │
│  result[Band1 > 0] = Band1 + offset                                │
│  result[Band2 > 0] = Band2 + offset                                │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  STAGE 5: VECTORIZATION                             │
│                                                                     │
│  • Raster labels → polygons                                        │
│  • Add attributes (area_m2, area_ha)                               │
│  • Export: Shapefile, GeoPackage, GeoJSON                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Base Tiles

### Tile Grid

The image is divided into non-overlapping tiles:

```python
tile_size = 2000  # pixels

# For a 15000×30000 image:
# Tiles in X: ceil(15000 / 2000) = 8
# Tiles in Y: ceil(30000 / 2000) = 15
# Total: 8 × 15 = 120 tiles
```

### Point Grid

For each tile, we generate a dense point grid for SAM:

```python
points_per_side = 64  # 64×64 = 4096 points per tile

# Points are normalized to [0, 1]
xs = np.linspace(0, 1, points_per_side)
ys = np.linspace(0, 1, points_per_side)
points = [[x, y] for y in ys for x in xs]
```

### SAM Inference

We use `SAM2AutomaticMaskGenerator` with custom point grids:

```python
generator = SAM2AutomaticMaskGenerator(
    model=sam_model,
    points_per_side=None,        # Disable default grid
    point_grids=[points],        # Use our custom grid
    pred_iou_thresh=0.88,
    stability_score_thresh=0.92,
    min_mask_region_area=1000,
    box_nms_thresh=0.7,
)

masks = generator.generate(tile_image)
```

---

## Stage 2: Cascade Refinement

### Concept

A single SAM pass may miss objects due to strict thresholds. We run multiple passes with progressively relaxed thresholds:

```
Pass 1:  iou=0.92, stability=0.95  → Easy objects
Pass 2:  iou=0.90, stability=0.93
Pass 3:  iou=0.88, stability=0.91
...
Pass 20: iou=0.56, stability=0.59  → Hard objects
```

### Black Masking

Between passes, already-segmented pixels are masked (set to black):

```python
for pass_idx in range(n_passes):
    # Black mask: already segmented areas
    masked_image = image.copy()
    masked_image[labels > 0] = 0  # RGB = (0, 0, 0)

    # SAM inference with current thresholds
    masks = generator.generate(masked_image)

    # Add new masks to labels
    for mask in masks:
        labels[mask['segmentation']] = next_label
        next_label += 1
```

### Threshold Interpolation

Thresholds are linearly interpolated:

```python
iou_start, iou_end = 0.92, 0.56
stability_start, stability_end = 0.95, 0.59

for i in range(n_passes):
    t = i / (n_passes - 1)  # 0 to 1
    iou = iou_start + t * (iou_end - iou_start)
    stability = stability_start + t * (stability_end - stability_start)
```

---

## Stage 3: Border Correction

### Discontinuity Detection

Discontinuities are the lines where tiles meet:

```python
tile_size = 2000

# Vertical discontinuities (X coordinates)
disc_v = [2000, 4000, 6000, 8000, ...]  # x = n × tile_size

# Horizontal discontinuities (Y coordinates)
disc_h = [2000, 4000, 6000, 8000, ...]  # y = n × tile_size
```

### V Tiles (Vertical Discontinuities)

For each vertical discontinuity, we create a tile centered on it:

```
        ←───── tile_size ─────→

       ┌──────────┬──────────┐
       │          │          │
       │    ·  ·  │  ·  ·    │   ↑
       │    ·  ·  │  ·  ·    │   │
       │    ·  ·  │  ·  ·    │   │  n_along = 100
       │    ·  ·  │  ·  ·    │   │  (points along discontinuity)
       │    ·  ·  │  ·  ·    │   │
       │    ·  ·  │  ·  ·    │   ↓
       │          │          │
       └──────────┴──────────┘
              ↑
         discontinuity

            ←─→
         n_across = 5
    (points across discontinuity)
```

Point grid generation:

```python
def make_v_grid(tile_size, zone_width, n_across, n_along):
    center_x = tile_size / 2
    margin = 20  # Edge margin

    # n_across columns in central zone
    xs = np.linspace(center_x - zone_width/2,
                     center_x + zone_width/2, n_across)

    # n_along rows distributed vertically
    ys = np.linspace(margin, tile_size - margin, n_along)

    # Normalize to [0, 1]
    points = [[x/tile_size, y/tile_size] for y in ys for x in xs]
    return np.array(points)
```

### H Tiles (Horizontal Discontinuities)

Same concept, rotated 90°:

```
       ┌─────────────────────────────┐
       │                             │
       │  · · · · · · · · · · · · ·  │  ↑ n_across = 5
       │  · · · · · · · · · · · · ·  │  ↓
       ├─────────────────────────────┤  ← discontinuity
       │  · · · · · · · · · · · · ·  │
       │  · · · · · · · · · · · · ·  │
       │                             │
       └─────────────────────────────┘

       ←─────────────────────────────→
              n_along = 100
```

### Corner Tiles

Corners are intersections of V and H discontinuities:

```
       ┌──────────┬──────────┐
       │          │          │
       │    · · · │ · · ·    │
       │    · · · │ · · ·    │  ← n_y = 5
       ├────· · ·─┼─· · ·────┤  ← H discontinuity
       │    · · · │ · · ·    │
       │    · · · │ · · ·    │
       │          │          │
       └──────────┴──────────┘
              ↑
         V discontinuity

            ←───→
            n_x = 5
```

### Mask Filtering

We only keep masks that **cross** the discontinuity:

```python
def crosses_v(mask, disc_x_local):
    """Check if mask crosses vertical discontinuity."""
    has_left = mask[:, :disc_x_local].any()
    has_right = mask[:, disc_x_local:].any()
    return has_left and has_right

def crosses_h(mask, disc_y_local):
    """Check if mask crosses horizontal discontinuity."""
    has_top = mask[:disc_y_local, :].any()
    has_bottom = mask[disc_y_local:, :].any()
    return has_top and has_bottom

# For corners: must cross BOTH
def crosses_corner(mask, disc_x_local, disc_y_local):
    return crosses_v(mask, disc_x_local) and crosses_h(mask, disc_y_local)
```

### Independent Cascades

**Important**: Each band starts from zero. Band 1 does NOT use Band 0 as a mask.

```python
# Band 0: Base tiles cascade (starts from zero)
band0 = run_cascade(base_tiles, ...)

# Band 1: V/H tiles cascade (starts from zero, independent)
band1 = run_cascade(vh_tiles, ...)

# Band 2: Corner tiles cascade (starts from zero, independent)
band2 = run_cascade(corner_tiles, ...)
```

This allows Band 1 and Band 2 to re-segment the fragmented objects from Band 0.

---

## Stage 4: Band Merge

### Priority-based Pixel Merge

```python
def merge_bands(band0, band1, band2):
    """Merge bands with priority: Band2 > Band1 > Band0."""

    max_b0 = band0.max()
    max_b1 = band1.max()

    result = band0.copy()

    # Band 1 overwrites Band 0
    result[band1 > 0] = band1[band1 > 0] + max_b0

    # Band 2 overwrites everything
    result[band2 > 0] = band2[band2 > 0] + max_b0 + max_b1

    return result
```

### Why This Priority?

- **Band 2 (corners)**: Objects at corners are most fragmented (split into 4 pieces). Corner tiles capture them whole.
- **Band 1 (V/H)**: Objects at edges are split into 2 pieces. V/H tiles capture them whole.
- **Band 0 (base)**: Fallback for objects not at boundaries.

---

## Stage 5: Vectorization

### Raster to Polygons

```python
from rasterio.features import shapes
from shapely.geometry import shape

def to_geodataframe(labels, transform, crs):
    """Convert label raster to GeoDataFrame."""
    mask = labels > 0
    polygons = []
    label_ids = []

    for geom, value in shapes(labels.astype(np.int32),
                               mask=mask,
                               transform=transform):
        if value > 0:
            polygons.append(shape(geom))
            label_ids.append(int(value))

    gdf = gpd.GeoDataFrame({
        'label_id': label_ids,
        'geometry': polygons
    }, crs=crs)

    return gdf
```

### Attribute Calculation

```python
def add_attributes(gdf):
    """Add area attributes."""
    gdf['area_m2'] = gdf.geometry.area
    gdf['area_ha'] = gdf['area_m2'] / 10000
    return gdf
```

---

## Summary

| Stage | Input | Output | Key Parameters |
|-------|-------|--------|----------------|
| Base Tiles | Image | Band 0 | tile_size, points_per_side |
| Cascade | Band 0 | Band 0 refined | n_passes, iou/stability range |
| V/H Tiles | Image | Band 1 | n_across, n_along, zone_width |
| Corners | Image | Band 2 | n_x, n_y |
| Merge | Bands 0,1,2 | Final labels | priority strategy |
| Vectorize | Final labels | Shapefile | format |

### Typical Results

| Metric | Base Only | + Border Correction |
|--------|-----------|---------------------|
| Coverage | 84.4% | **94.0%** |
| Fragments at borders | Many | Few |
| Processing time | ~6.5h | +11min |
