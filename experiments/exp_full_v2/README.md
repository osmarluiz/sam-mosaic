# SAM-Mosaic Experiment: Full Image Segmentation

## Quick Start for Colleagues

Copy this entire message to Claude and provide your image path:

---

**Para Claude:**

Execute o pipeline SAM-Mosaic no seguinte arquivo de imagem:

```
[COLOQUE AQUI O CAMINHO COMPLETO DA SUA IMAGEM]
```

Exemplo: `D:/dados/minha_imagem.tif`

### Instruções:

1. Edite o arquivo `experiments/exp_full_v2/run.py` e altere a linha `SOURCE_IMAGE`:

```python
SOURCE_IMAGE = Path("CAMINHO_DA_SUA_IMAGEM_AQUI")
```

2. Verifique se o checkpoint do SAM está correto em `experiments/exp_full_v2/config.yaml`:

```yaml
sam:
  checkpoint: "CAMINHO_DO_CHECKPOINT_SAM2"  # ex: D:/SAM2/checkpoints/sam2.1_hiera_large.pt
```

3. Execute:

```cmd
python D:\sam-mosaic\experiments\exp_full_v2\run.py
```

---

## Configuration Details

### Parameters (config.yaml)

| Stage | Passes | Points (Pass 0) | Points (Pass 1+) | IoU/Stability |
|-------|--------|-----------------|------------------|---------------|
| Base Tiles | 15 | 64x64 = 4096 | K-means 64 pts | 0.92 → 0.78 |
| V/H Borders | 5 | 3x80 = 240 | K-means 64 pts (zone) | 0.92 → 0.88 |
| Corners | 5 | 3x3 = 9 | K-means 9 pts (zone) | 0.92 → 0.88 |

### Post-processing (Base Tiles)
- Remove masks < 100 pixels
- Merge enclosed masks < 500 pixels
- Edge completion: search up to 15 pixels towards tile center

### Tiling
- Tile size: 2000x2000 pixels
- Zone width: 50 pixels (for border correction)

## Output Files

```
output/
├── band0_base.tif           # Base tiles (original)
├── band0_base_processed.tif # Base tiles (post-processed)
├── band1_vh.tif             # V/H border corrections
├── band2_corners.tif        # Corner corrections
├── merged_labels.tif        # Final merged result
├── segments.shp             # Vectorized polygons
└── state/                   # Intermediate state files
```

## Requirements

- Python 3.10+
- SAM2 checkpoint (sam2.1_hiera_large.pt)
- CUDA GPU recommended
- ~16GB RAM for large images

## Dependencies

```bash
pip install torch torchvision
pip install rasterio shapely geopandas
pip install scikit-learn scipy tqdm pyyaml
```
