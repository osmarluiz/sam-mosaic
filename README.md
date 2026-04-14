# Remote SAMsing

**From Segment Anything to Segment Everything**

Segment large remote sensing images (satellite, aerial, drone) that exceed GPU memory by processing them in tiles and intelligently merging the results. Built on SAM2, the `sam-mosaic` package achieves 91--98% coverage across diverse scenes without any model fine-tuning or manual annotation.

Tested on 7 scenes spanning 5 cm to 4.78 m GSD, two spectral compositions (natural RGB and MNF false-color), and two landscape types (urban and agricultural), including a scalability test on a 36,000 x 54,000 pixel mosaic (1.94 billion pixels).

> **Paper:** O. L. F. de Carvalho, O. A. de Carvalho Junior, A. O. de Albuquerque, and D. Guerreiro e Silva, "Remote SAMsing: From Segment Anything to Segment Everything," *Int. J. Applied Earth Observation and Geoinformation*, 2026.

## Features

- **Multi-pass segmentation** with adaptive thresholds for high coverage (91--98%)
- **Black mask focusing** to direct SAM toward residual unsegmented areas
- **Best-match boundary merge** at tile edges using LUT + Union-Find (parameter-free, O(n) complexity)
- **Dense Grid point strategy** (default) for robust performance across object scales
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

> **Important - SAM2 Version**: This project uses `sam2` from PyPI ([JinsuaFeito-dev fork](https://github.com/JinsuaFeito-dev/segment-anything-2)), NOT the official Facebook repository. This version (1.1.0+) has been tested for stable GPU memory usage during large-scale processing. Do NOT install from `pip install git+https://github.com/facebookresearch/sam2.git` as it may cause memory leaks.

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
| `--point-strategy` | dense_grid | Point selection: `dense_grid` (default, robust) or `kmeans` (alternative). |
| `--erosion` | 0 | Erosion iterations for point placement. Increase for denser scenes. |
| `--iou-start` | 0.93 | Initial IoU threshold (restrictive). |
| `--iou-end` | 0.60 | Final IoU threshold (permissive). |
| `--stability-start` | 0.93 | Initial stability score threshold. |
| `--stability-end` | 0.60 | Final stability score threshold. |

### Point Strategies

Remote SAMsing supports two point selection strategies for multi-pass segmentation:

**Dense Grid (default)**: Uses a uniform grid filtered by already-segmented areas. Robust across scene types, from urban imagery with small objects to agricultural fields with large parcels.

**K-means**: Clusters points in residual (unsegmented) areas. An alternative when targeting large, homogeneous regions.

```bash
# Default: Dense Grid (works well for most scenes)
sam-mosaic input.tif output/ --checkpoint sam2.pt

# Dense Grid with higher point density for very small objects
sam-mosaic input.tif output/ --checkpoint sam2.pt \
    --points-per-side 96

# K-means for large homogeneous regions
sam-mosaic input.tif output/ --checkpoint sam2.pt \
    --point-strategy kmeans --erosion 5
```

### Threshold Parameters

SAM2 uses two thresholds to filter predicted masks:

- **IoU threshold** (`--iou-start`, `--iou-end`): Filters masks by predicted IoU score
- **Stability threshold** (`--stability-start`, `--stability-end`): Filters masks by stability score

Both thresholds decrease from start to end across passes, allowing more permissive masks as coverage increases. Strict thresholds capture salient objects first; relaxation occurs only when progress stagnates.

```bash
# More restrictive (fewer but higher quality segments)
sam-mosaic input.tif output/ --checkpoint sam2.pt \
    --iou-start 0.95 --stability-start 0.95

# More permissive (higher coverage, may include lower quality segments)
sam-mosaic input.tif output/ --checkpoint sam2.pt \
    --iou-end 0.50 --stability-end 0.50
```

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
  point_strategy: dense_grid
  points_per_side: 64  # Grid density (64x64 = 4096 points)
  target_coverage: 99.0
  use_black_mask: true
  use_adaptive_threshold: true

merge:
  merge_strategy: best_match
  min_contact_pixels: 20
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

Remote SAMsing processes each tile in multiple passes with decreasing quality thresholds:

1. **Pass 1**: Uniform grid (64x64 = 4096 points) with strict IoU/stability thresholds (0.93). Captures high-confidence segments first (~60--70% coverage).

2. **Pass 2+**: Points placed only in residual (unsegmented) areas via the Dense Grid strategy. A black mask is applied to already-segmented pixels, directing SAM toward remaining gaps. Thresholds decrease gradually (0.93 -> 0.92 -> ... -> 0.60), but only when coverage progress stagnates.

3. **Stop condition**: Coverage >= 99% or minimum threshold reached.

After all tiles are processed, segments touching at tile boundaries are reconciled through a **best-match merge**: each label pair at a discontinuity line is scored by contact length, and the best match for each label is accepted. Union-Find resolves transitive chains, and a single LUT lookup relabels the full image in O(n) time. This merge is parameter-free and produces a spatially consistent label map for arbitrarily large images with constant GPU memory.

---

## Requirements

- **Python**: 3.12
- **GPU**: NVIDIA GPU with CUDA (recommended). Works on CPU but much slower.
- **RAM**: 16GB+ recommended for large images
- **VRAM**: 8GB+ recommended (tested with 24GB GPU on images up to 1.94 billion pixels)
- **Disk**: ~1GB for SAM2 checkpoint + space for outputs

### Dependencies

- PyTorch 2.9
- SAM2 1.1.0+ (from PyPI)
- CUDA 12.8
- rasterio
- numpy, scipy, scikit-learn
- shapely, fiona
- tqdm, pyyaml

### Tested Configuration

| Component | Specification |
|-----------|---------------|
| CPU | Intel Core i9-14900K |
| RAM | 64 GB |
| GPU | NVIDIA RTX 4090 (24 GB VRAM) |
| Python | 3.12 |
| PyTorch | 2.9 |
| SAM2 | 1.1.0 |
| CUDA | 12.8 |

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

# Without black mask
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

# K-means point strategy (alternative to default Dense Grid)
result = segment_with_params(
    "input.tif", "output/kmeans/",
    checkpoint="checkpoints/sam2.1_hiera_large.pt",
    point_strategy="kmeans",
    erosion_iterations=5
)

# Custom stability thresholds
result = segment_with_params(
    "input.tif", "output/custom_thresholds/",
    checkpoint="checkpoints/sam2.1_hiera_large.pt",
    stability_start=0.90,
    stability_end=0.50
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
@article{carvalho2026remotesamsing,
  title = {Remote {SAMsing}: From Segment Anything to Segment Everything},
  author = {de Carvalho, Osmar Luiz Ferreira and de Carvalho J{\'u}nior, Osmar Ab{\'i}lio and de Albuquerque, Anesmar Olino and Guerreiro e Silva, Daniel},
  journal = {International Journal of Applied Earth Observation and Geoinformation},
  year = {2026},
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
