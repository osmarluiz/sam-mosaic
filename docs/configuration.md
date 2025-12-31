# Configuration Reference

This document describes all configuration parameters for SAM-Mosaic.

## Table of Contents

1. [Input](#input)
2. [Tiling](#tiling)
3. [SAM Model](#sam-model)
4. [Base Tiles](#base-tiles)
5. [Border Correction](#border-correction)
6. [Merge](#merge)
7. [Output](#output)

---

## Input

```yaml
input:
  image_path: "data/image.tif"
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image_path` | string | Yes | Path to input image |

### Supported Formats

| Format | Extensions | Geospatial |
|--------|------------|------------|
| PNG | .png | No |
| JPEG | .jpg, .jpeg | No |
| TIFF | .tif, .tiff | Optional |
| GeoTIFF | .tif, .tiff | Yes |
| ENVI | .hdr + .dat | Yes |

---

## Tiling

```yaml
tiling:
  tile_size: 2000
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tile_size` | int | 2000 | Tile size in pixels |

### Guidelines

| GPU VRAM | Recommended tile_size |
|----------|----------------------|
| 8 GB | 1024 |
| 12 GB | 1500 |
| 16 GB | 2000 |
| 24 GB+ | 2500 |

---

## SAM Model

```yaml
sam:
  model_config: "configs/sam2.1/sam2.1_hiera_l.yaml"
  checkpoint: "models/sam2.1_hiera_large.pt"
  device: "cuda"
  min_mask_area: 1000
  box_nms_thresh: 0.7
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_config` | string | Required | Path to SAM config YAML |
| `checkpoint` | string | Required | Path to SAM checkpoint |
| `device` | string | "cuda" | Device: "cuda" or "cpu" |
| `min_mask_area` | int | 1000 | Minimum mask area in pixels |
| `box_nms_thresh` | float | 0.7 | NMS threshold for boxes |

### Available Models

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| sam2.1_hiera_tiny | 39M | Fast | Good |
| sam2.1_hiera_small | 46M | Fast | Good |
| sam2.1_hiera_base_plus | 80M | Medium | Better |
| sam2.1_hiera_large | 224M | Slow | Best |

---

## Base Tiles

```yaml
base_tiles:
  grid:
    points_per_side: 64

  cascade:
    n_passes: 20
    thresholds:
      iou: [0.92, 0.56]
      stability: [0.95, 0.59]
```

### Grid Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `points_per_side` | int | 64 | Points per side (total = n²) |

**Total points per tile**: `points_per_side × points_per_side`

| points_per_side | Total points |
|-----------------|--------------|
| 32 | 1,024 |
| 64 | 4,096 |
| 128 | 16,384 |

### Cascade Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_passes` | int | 20 | Number of cascade iterations |
| `thresholds.iou` | [float, float] | [0.92, 0.56] | IoU threshold [start, end] |
| `thresholds.stability` | [float, float] | [0.95, 0.59] | Stability threshold [start, end] |

**Threshold interpolation**: Linear from start to end over n_passes.

```
Pass 1:  iou = 0.92, stability = 0.95
Pass 10: iou = 0.74, stability = 0.77
Pass 20: iou = 0.56, stability = 0.59
```

---

## Border Correction

```yaml
border_correction:
  zone_width: 100

  v_tiles:
    grid:
      n_across: 5
      n_along: 100
    cascade:
      n_passes: 5
      thresholds:
        iou: [0.88, 0.70]
        stability: [0.92, 0.75]

  h_tiles:
    grid:
      n_across: 5
      n_along: 100
    cascade:
      n_passes: 5
      thresholds:
        iou: [0.88, 0.70]
        stability: [0.92, 0.75]

  corner_tiles:
    grid:
      n_x: 5
      n_y: 5
    cascade:
      n_passes: 5
      thresholds:
        iou: [0.88, 0.70]
        stability: [0.92, 0.75]
```

### Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `zone_width` | int | 100 | Width of point concentration zone |

### V/H Tiles Grid

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_across` | int | Points crossing the discontinuity |
| `n_along` | int | Points along the discontinuity |

**V tiles**: Points form a vertical strip.
```
n_across = 5 columns (in center)
n_along = 100 rows (full height)
Total = 500 points
```

**H tiles**: Points form a horizontal strip.
```
n_across = 5 rows (in center)
n_along = 100 columns (full width)
Total = 500 points
```

### Corner Tiles Grid

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_x` | int | Columns in the grid |
| `n_y` | int | Rows in the grid |

```
n_x = 5, n_y = 5
Total = 25 points (centered on corner)
```

### Border Cascade

Same as base tiles cascade, but typically fewer passes:

| Tile Type | Recommended n_passes |
|-----------|---------------------|
| Base | 20 |
| V/H | 5 |
| Corner | 5 |

---

## Merge

```yaml
merge:
  strategy: "priority"
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `strategy` | string | "priority" | Merge strategy |

### Strategies

| Strategy | Description |
|----------|-------------|
| `priority` | Band2 > Band1 > Band0 (recommended) |

**Priority merge logic**:
```python
result = band0
result[band1 > 0] = band1 + offset
result[band2 > 0] = band2 + offset
```

---

## Output

```yaml
output:
  dir: "output/"
  save_bands: true
  save_passes: false

  vectorize:
    enabled: true
    format: "shapefile"
    simplify_tolerance: 0
```

### General Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dir` | string | "output/" | Output directory |
| `save_bands` | bool | true | Save individual bands |
| `save_passes` | bool | false | Save each cascade pass |

### Vectorize Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | true | Enable vectorization |
| `format` | string | "shapefile" | Output format |
| `simplify_tolerance` | float | 0 | Simplification tolerance (0 = none) |

### Output Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| `shapefile` | .shp | ESRI Shapefile |
| `geopackage` | .gpkg | OGC GeoPackage |
| `geojson` | .geojson | GeoJSON |

### Output Files

With `save_bands: true`:

```
output/
├── band0.tif           # Base tiles result
├── band1.tif           # V/H tiles result
├── band2.tif           # Corner tiles result
├── final.tif           # Merged result
├── final.shp           # Vectorized (if enabled)
├── final.dbf
├── final.shx
├── final.prj
└── summary.txt         # Processing summary
```

---

## Complete Example

```yaml
input:
  image_path: "data/Plant23_NDVI_MNF.tif"

tiling:
  tile_size: 2000

sam:
  model_config: "configs/sam2.1/sam2.1_hiera_l.yaml"
  checkpoint: "models/sam2.1_hiera_large.pt"
  device: "cuda"
  min_mask_area: 1000
  box_nms_thresh: 0.7

base_tiles:
  grid:
    points_per_side: 64
  cascade:
    n_passes: 20
    thresholds:
      iou: [0.92, 0.56]
      stability: [0.95, 0.59]

border_correction:
  zone_width: 100

  v_tiles:
    grid:
      n_across: 5
      n_along: 100
    cascade:
      n_passes: 5
      thresholds:
        iou: [0.88, 0.70]
        stability: [0.92, 0.75]

  h_tiles:
    grid:
      n_across: 5
      n_along: 100
    cascade:
      n_passes: 5
      thresholds:
        iou: [0.88, 0.70]
        stability: [0.92, 0.75]

  corner_tiles:
    grid:
      n_x: 5
      n_y: 5
    cascade:
      n_passes: 5
      thresholds:
        iou: [0.88, 0.70]
        stability: [0.92, 0.75]

merge:
  strategy: "priority"

output:
  dir: "output/plant23/"
  save_bands: true
  save_passes: false
  vectorize:
    enabled: true
    format: "shapefile"
    simplify_tolerance: 0
```
