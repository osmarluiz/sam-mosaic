# MethodsX Specs — Remote SAMsing

## Meta
- Journal: MethodsX (Elsevier)
- Type: Method article
- Companion to: JAG paper "Remote SAMsing: From Segment Anything to Segment Everything"
- No word limit for Method Details
- Max 25 references
- No Introduction section (starts with Abstract → Method Details)

## Title
Remote SAMsing: A Python Pipeline for SAM2-Based Segmentation of Large Remote Sensing Images

## Abstract (max 500 words)
- What the method does (multi-pass SAM2 pipeline for large RS images)
- Why it's needed (SAM2 alone leaves 30-70% unsegmented, no boundary merge)
- How to use it (pip install, CLI or Python API)
- Key capabilities (arbitrarily large images, constant GPU memory, no training)
- Reference to companion JAG paper for scientific evaluation

## Method Details (core)
### 1. Installation and Requirements
- Python 3.12+, PyTorch 2.x, CUDA
- SAM2 checkpoint download
- pip install from GitHub
- Hardware requirements (GPU with 8GB+ VRAM)

### 2. Pipeline Overview
- Brief recap of the algorithm (reference JAG paper)
- Flowchart: Input → Tiling → Multi-pass per tile → Merge → Output

### 3. Usage via Python API
- `segment_with_params()` — full parameter control
- `segment()` — config file based
- Code example: minimal 3-line usage
- Code example: custom parameters for urban scene
- Code example: agricultural scene with different tile size

### 4. Configuration Parameters
- Table with ALL parameters, defaults, ranges, and guidance
- When to change tile_size (scale parameter)
- When to change thresholds
- When to enable crop_n_layers
- streaming_mode: ram vs disk (memory considerations)

### 5. Output Formats
- GeoTIFF labels raster (georeferenced)
- Shapefile / GeoPackage (vector polygons)
- Simplification tolerance
- Integration with QGIS, ArcGIS, Python (geopandas)

### 6. CLI Usage
- Command-line interface
- Config YAML file format
- Batch processing multiple images

### 7. Processing Large Images
- Tile-by-tile processing for memory efficiency
- Streaming mode (disk vs RAM)
- GPU memory management (reset intervals)
- Expected processing times per image size

### 8. Practical Recommendations
- Tile size selection guide (50m minimum ground coverage)
- Urban scenes: T=250 for small objects
- Agricultural: T=1000 for large fields
- Mixed: start with T=1000, reduce if needed

## Method Validation
- Reference JAG paper Tables 3-6 for quantitative validation
- Show one example: BSB-1 input → output with coverage/segments stats
- Show one example: Potsdam full mosaic (1.94B pixels)

## Limitations
- Processing time (~18h for 8k×8k at T=250)
- SAM2 dependency (requires specific checkpoint)
- RGB input only (SAR, thermal untested)
- No class labels (unsupervised segmentation)

## References
- JAG companion paper (Remote SAMsing)
- SAM2 (Ravi 2024)
- segment-geospatial (Wu 2023)
- SLIC (Achanta 2012)
- Felzenszwalb (2004)
- Union-Find (Tarjan 1975)
- ~5-10 more as needed (max 25 total)
