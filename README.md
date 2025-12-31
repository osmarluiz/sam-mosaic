# SAM-Mosaic

Segment large images using SAM 2.1 with tile-based processing and automatic border correction.

## Overview

SAM-Mosaic solves the problem of segmenting large images (e.g., 15000×30000 pixels) that cannot fit in GPU memory. It divides the image into tiles, runs SAM on each tile, and automatically fixes segmentation artifacts at tile boundaries.

### Key Features

- **Tile-based processing**: Configurable tile size for any GPU memory
- **Cascade refinement**: Multiple passes with decreasing thresholds
- **Border correction**: Automatic fixing of objects split across tile boundaries
- **Multi-format support**: PNG, JPEG, TIFF, GeoTIFF, ENVI
- **Geospatial output**: Preserves CRS and exports to Shapefile/GeoPackage

## Installation

```bash
git clone https://github.com/yourusername/sam-mosaic.git
cd sam-mosaic
pip install -e .
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- SAM 2.1
- rasterio
- geopandas
- numpy

## Quick Start

```bash
# Run with config file
python scripts/run_pipeline.py --config configs/example.yaml

# Or with command line args
python scripts/run_pipeline.py --image data/image.tif --output output/ --tile-size 2000
```

## How It Works

### Pipeline Overview

```
Input Image (large)
       │
       ▼
┌──────────────────┐
│  1. BASE TILES   │  Divide into tiles, run SAM with cascade
│     → Band 0     │
└──────────────────┘
       │
       ▼
┌──────────────────┐
│  2. V/H TILES    │  Re-segment at vertical/horizontal boundaries
│     → Band 1     │
└──────────────────┘
       │
       ▼
┌──────────────────┐
│  3. CORNERS      │  Re-segment at tile corners
│     → Band 2     │
└──────────────────┘
       │
       ▼
┌──────────────────┐
│  4. MERGE        │  Combine bands: Band2 > Band1 > Band0
└──────────────────┘
       │
       ▼
Output (raster + vector)
```

### The Border Problem

When processing tiles independently, objects at tile boundaries get fragmented:

```
    Tile A    │    Tile B
              │
      ┌───────┼───────┐
      │   3a  │  3b   │   ← Object split into fragments!
      └───────┼───────┘
              │
```

### The Solution: 3-Band NMS

We re-segment areas around tile boundaries using centered tiles:

- **Band 0**: Base tile segmentation (has fragments at borders)
- **Band 1**: V/H tiles centered on discontinuities (captures crossing objects)
- **Band 2**: Corner tiles at V+H intersections (highest priority)

Merge with priority: `Band2 > Band1 > Band0`

### Cascade Refinement

Each band uses multiple passes with decreasing thresholds:

```
Pass 1: iou=0.92, stability=0.95  (strict - easy objects)
Pass 2: iou=0.88, stability=0.91  (slightly relaxed)
...
Pass N: iou=0.56, stability=0.59  (relaxed - hard objects)
```

Between passes, already-segmented pixels are masked (black), so SAM focuses on remaining gaps.

## Configuration

See [docs/configuration.md](docs/configuration.md) for full reference.

### Example Config

```yaml
input:
  image_path: "data/image.tif"

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
    points_per_side: 64          # 64×64 = 4096 points
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
      n_along: 100               # 500 points
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
      n_y: 5                     # 25 points
    cascade:
      n_passes: 5
      thresholds:
        iou: [0.88, 0.70]
        stability: [0.92, 0.75]

merge:
  strategy: "priority"

output:
  dir: "output/"
  save_bands: true
  vectorize:
    enabled: true
    format: "shapefile"
```

## Project Structure

```
sam-mosaic/
├── README.md
├── LICENSE
├── pyproject.toml
│
├── docs/
│   ├── methodology.md          # Detailed methodology
│   └── configuration.md        # Config reference
│
├── configs/
│   └── example.yaml            # Example configuration
│
├── src/sam_mosaic/
│   ├── config.py               # Config loader
│   ├── pipeline.py             # Main orchestrator
│   │
│   ├── io/
│   │   ├── loader.py           # Image loading
│   │   └── writer.py           # Raster saving
│   │
│   ├── tiling/
│   │   ├── grid.py             # Tile grid generation
│   │   └── borders.py          # Discontinuity detection
│   │
│   ├── points/
│   │   └── grids.py            # Point grid generation
│   │
│   ├── segmentation/
│   │   ├── sam.py              # SAM wrapper
│   │   └── cascade.py          # Cascade logic
│   │
│   ├── merge/
│   │   └── bands.py            # Band merging
│   │
│   └── vectorize/
│       └── polygonize.py       # Raster to vector
│
├── scripts/
│   └── run_pipeline.py         # CLI entry point
│
└── examples/
    └── quick_start.ipynb       # Tutorial notebook
```

## API Reference

### Core Classes

```python
from sam_mosaic import Pipeline
from sam_mosaic.config import load_config

# Load config and run
cfg = load_config("config.yaml")
pipeline = Pipeline(cfg)
result = pipeline.run()
```

### Individual Components

```python
from sam_mosaic.io import load_image, save_raster
from sam_mosaic.tiling import TileGrid
from sam_mosaic.points import make_grid, make_v_grid
from sam_mosaic.segmentation import SAMWrapper, Cascade
from sam_mosaic.merge import merge_bands
from sam_mosaic.vectorize import to_geodataframe, save_shapefile
```

## Results

Tested on a 15000×30000 pixel agricultural image:

| Stage | Coverage | Labels | Time |
|-------|----------|--------|------|
| Base tiles (20 passes) | 84.4% | 14,096 | ~6.5h |
| + Border correction | **94.0%** | 15,841 | +11min |

## Citation

If you use this work, please cite:

```bibtex
@software{sam_mosaic,
  title = {SAM-Mosaic: Large Image Segmentation with SAM},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/sam-mosaic}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
