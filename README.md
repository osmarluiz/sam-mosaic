# SAM-Mosaic

Large-scale image segmentation using SAM2 with intelligent tiling and merging.

Segment large remote sensing images (satellite, aerial, drone) that don't fit in GPU memory by processing them in tiles and intelligently merging the results.

## Features

- **Multi-pass segmentation** with adaptive thresholds for high coverage (>99%)
- **Black mask focusing** to help SAM segment residual areas efficiently
- **Optimized merge** at tile boundaries using LUT + Union-Find (O(n) complexity)
- **K-means point placement** in residual areas for better distribution
- **Adaptive tile padding** to ensure clean merges at boundaries
- **GeoTIFF support** with CRS and georeferencing preservation
- **Multiple output formats**: Raster labels (TIFF) + Vector polygons (Shapefile/GeoPackage)

---

## Quick Start

### 1. Install

**Option A: Using conda (recommended for exact reproducibility)**

```bash
# Clone the repository
git clone https://github.com/osmarluiz/sam-mosaic.git
cd sam-mosaic

# Create conda environment from file
conda env create -f environment.yml
conda activate ts_annotator
```

**Option B: Manual installation**

```bash
# Clone the repository
git clone https://github.com/osmarluiz/sam-mosaic.git
cd sam-mosaic

# Install the package
pip install -e .

# Install SAM2 from PyPI (recommended)
pip install sam2
```

> **Important - SAM2 Version**: We use `sam2` from PyPI ([JinsuaFeito-dev fork](https://github.com/JinsuaFeito-dev/segment-anything-2)), NOT the official Facebook repository. This version (1.1.0+) has been tested for stable GPU memory usage during large-scale processing. Do NOT install from `pip install git+https://github.com/facebookresearch/sam2.git` as it may cause memory leaks.

### 2. Download SAM2 Model

Download the SAM2 checkpoint (~857MB):

```bash
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
cd ..
```

> See `checkpoints/README.md` for other model sizes (tiny, small, base).

### 3. Run Segmentation

```bash
# Run from inside the sam-mosaic directory (uses default config)
sam-mosaic /path/to/your/image.tif /path/to/output/

# Example (run from sam-mosaic folder)
sam-mosaic /data/ortofoto.tif /results/segmentation/

# Or specify checkpoint explicitly (can run from anywhere)
sam-mosaic /path/to/image.tif /path/to/output/ --checkpoint /path/to/checkpoints/sam2.1_hiera_large.pt
```

That's it! The tool will generate:
- `labels.tif` - Raster with segment labels
- `segments.shp` - Vectorized polygons (Shapefile)
- `stats.json` - Processing statistics

---

## Usage

### Command Line (CLI)

```bash
# Basic usage (uses optimized default parameters)
sam-mosaic input.tif output/

# With custom checkpoint path
sam-mosaic input.tif output/ --checkpoint /path/to/sam2.1_hiera_large.pt

# With custom configuration file
sam-mosaic input.tif output/ --config my_config.yaml

# Customize polygon simplification (default: 1.0)
sam-mosaic input.tif output/ --simplify-tolerance 2.0

# Also generate GeoPackage
sam-mosaic input.tif output/ --geopackage
```

### Python API

```python
from sam_mosaic import segment_image

# Basic usage
result = segment_image(
    input_path="data/ortofoto.tif",
    output_dir="results/",
)

print(f"Segments found: {result.n_segments}")
print(f"Coverage: {result.coverage:.1f}%")
print(f"Processing time: {result.processing_time:.1f}s")
print(f"Labels saved to: {result.labels_path}")
print(f"Shapefile saved to: {result.shapefile_path}")
```

---

## Output Files

After running, the output directory will contain:

| File | Description |
|------|-------------|
| `labels.tif` | GeoTIFF raster where each pixel has a segment ID (1, 2, 3, ...). Background = 0. Preserves CRS and georeferencing from input. |
| `segments.shp` | Shapefile with vectorized polygons. Includes attributes: `label_id`, `area_m2`, `perimeter_m`. |
| `segments.gpkg` | GeoPackage (optional, use `--geopackage` flag) |
| `stats.json` | Detailed statistics: segments count, coverage, processing time, per-tile stats. |

---

## Customization

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--simplify-tolerance` | 1.0 | Polygon simplification in map units. Higher = simpler polygons. Use 0 for no simplification. |
| `--tile-size` | 1000 | Tile size in pixels. Reduce if running out of GPU memory. |
| `--padding` | 50 | Extra context pixels around each tile. Helps with boundary merging. |
| `--min-area` | 100 | Remove segments smaller than this (in pixels). |
| `--target-coverage` | 99.0 | Stop segmentation when this coverage % is reached. |

### Polygon Simplification Examples

```bash
# No simplification (keeps all vertices - larger file)
sam-mosaic input.tif output/ --simplify-tolerance 0

# Light simplification (default)
sam-mosaic input.tif output/ --simplify-tolerance 1.0

# More simplification (smaller file, less detail)
sam-mosaic input.tif output/ --simplify-tolerance 3.0

# Heavy simplification
sam-mosaic input.tif output/ --simplify-tolerance 5.0
```

### Using a Configuration File

Create a `config.yaml` file:

```yaml
tile:
  size: 1000           # Tile size in pixels
  padding: 50          # Context padding

threshold:
  iou_start: 0.93      # Initial IoU threshold (restrictive)
  iou_end: 0.60        # Final IoU threshold (permissive)
  step: 0.01           # Threshold decrease per pass

segmentation:
  points_per_side: 64  # Grid density (64x64 = 4096 points)
  target_coverage: 99.0
  use_black_mask: true
  use_adaptive_threshold: true

merge:
  min_contact_pixels: 5
  min_mask_area: 100
  merge_enclosed_max_area: 500

output:
  simplify_tolerance: 1.0
  save_shapefile: true
  save_geopackage: false

sam_checkpoint: checkpoints/sam2.1_hiera_large.pt
```

Then run:

```bash
sam-mosaic input.tif output/ --config config.yaml
```

---

## How It Works

### Multi-Pass Segmentation

The algorithm processes each tile in multiple passes:

1. **Pass 1**: Uniform grid (64x64 = 4096 points) with high IoU threshold (0.93)
   - Gets high-confidence segments first (~60-70% coverage)

2. **Pass 2+**: K-means points in residual (unsegmented) areas
   - Black mask applied to already-segmented areas
   - Threshold decreases gradually: 0.93 → 0.92 → ... → 0.60
   - Focuses SAM on remaining difficult areas

3. **Stop condition**: Coverage ≥ 99% or minimum threshold reached

### Tile Processing with Padding

```
┌────────────────────────────────────┐
│  Padding (50px)                    │
│  ┌──────────────────────────────┐  │
│  │                              │  │
│  │      Useful Area             │  │
│  │      (1000x1000)             │  │
│  │                              │  │
│  └──────────────────────────────┘  │
│                                    │
└────────────────────────────────────┘
        Total read: 1100x1100
```

Padding provides context at tile boundaries, ensuring segments extend to edges for proper merging.

### Optimized Boundary Merge

After all tiles are processed, segments touching at tile boundaries are merged:

1. Find all label pairs touching at discontinuity lines
2. Filter by minimum contact (default: 5 pixels)
3. Use Union-Find for transitive merges
4. Apply with single LUT lookup (O(n) complexity)

---

## Requirements

- **Python**: 3.10+
- **GPU**: NVIDIA GPU with CUDA (recommended). Works on CPU but much slower.
- **RAM**: 16GB+ recommended for large images
- **VRAM**: 8GB+ recommended (tested with 24GB GPU on 10k×10k images)
- **Disk**: ~1GB for SAM2 checkpoint + space for outputs

### Dependencies

- PyTorch 2.0+
- SAM2 1.1.0+ (from PyPI)
- rasterio
- numpy, scipy, scikit-learn
- shapely, fiona
- tqdm, pyyaml

### Tested Configuration

| Component | Version |
|-----------|---------|
| Python | 3.10, 3.11 |
| SAM2 | 1.1.0 |
| PyTorch | 2.0+ |
| CUDA | 12.6 |
| NVIDIA Driver | 560.94 |

---

## Ablation Studies

The package supports easy ablation experiments:

```python
from sam_mosaic import segment_with_params

# Single-pass only (no multi-pass)
result = segment_with_params(
    "input.tif", "output/single_pass/",
    checkpoint="checkpoints/sam2.1_hiera_large.pt",
    max_passes=1,
    iou_start=0.86,
    use_adaptive_threshold=False
)

# Without black mask (5-10x slower)
result = segment_with_params(
    "input.tif", "output/no_blackmask/",
    checkpoint="checkpoints/sam2.1_hiera_large.pt",
    use_black_mask=False
)

# Without padding (to show merge artifacts)
result = segment_with_params(
    "input.tif", "output/no_padding/",
    checkpoint="checkpoints/sam2.1_hiera_large.pt",
    padding=0
)
```

---

## Troubleshooting

### CUDA out of memory

Reduce tile size:

```bash
sam-mosaic input.tif output/ --tile-size 512
```

### Too many small segments

Increase minimum area filter:

```bash
sam-mosaic input.tif output/ --min-area 200
```

### Shapefile too large / too many vertices

Increase simplification:

```bash
sam-mosaic input.tif output/ --simplify-tolerance 3.0
```

### Segments not merging at tile boundaries

Increase padding:

```bash
sam-mosaic input.tif output/ --padding 100
```

### OpenMP library conflict (Windows)

If you see an error about `libomp.dll` and `libiomp5md.dll` conflict:

```powershell
# PowerShell - set before running
$env:KMP_DUPLICATE_LIB_OK='TRUE'
sam-mosaic input.tif output/
```

```bash
# Bash/CMD
set KMP_DUPLICATE_LIB_OK=TRUE
sam-mosaic input.tif output/
```

### CLI hangs at "Loading SAM2 model..."

Enable debug mode to identify where it hangs:

```powershell
$env:SAM_MOSAIC_DEBUG='1'
$env:KMP_DUPLICATE_LIB_OK='TRUE'
sam-mosaic input.tif output/ --checkpoint path/to/sam2.pt
```

This will print detailed loading steps. The last `[DEBUG]` message before hanging indicates the problem.

---

## Citation

If you use this software in your research, please cite:

```bibtex
@software{sam_mosaic,
  title = {SAM-Mosaic: Large-scale Image Segmentation with SAM2},
  author = {Carvalho, Osmar Luiz Ferreira de},
  year = {2025},
  url = {https://github.com/osmarluiz/sam-mosaic}
}
```

---

## License

MIT License

---

## Acknowledgments

- [SAM2](https://github.com/facebookresearch/sam2) by Meta AI - Segment Anything Model 2
- [sam2 PyPI package](https://pypi.org/project/sam2/) - SAM2 distribution used in this project
- [rasterio](https://rasterio.readthedocs.io/) for GeoTIFF handling
- [shapely](https://shapely.readthedocs.io/) and [fiona](https://fiona.readthedocs.io/) for vector operations
