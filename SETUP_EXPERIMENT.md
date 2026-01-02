# Guia para Executar o Experimento SAM-Mosaic

Este documento explica como configurar e executar o experimento de segmentação SAM-Mosaic em um novo computador.

## Pré-requisitos

- Windows 10/11 com WSL2 ou Linux
- Python 3.10+
- CUDA 11.8+ (GPU NVIDIA com pelo menos 8GB VRAM)
- Anaconda ou Miniconda
- Git

## Passo 1: Clonar o Repositório

```bash
git clone https://github.com/osmarluiz/sam-mosaic.git
cd sam-mosaic
```

## Passo 2: Criar Ambiente Conda

```bash
conda create -n sam_mosaic python=3.11 -y
conda activate sam_mosaic
```

## Passo 3: Instalar PyTorch com CUDA

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Passo 4: Instalar SAM 2.1

```bash
pip install sam2
```

Ou clonar o repositório oficial:

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
cd ..
```

## Passo 5: Baixar o Checkpoint do SAM 2.1

Baixar o modelo `sam2.1_hiera_large.pt` de:
https://github.com/facebookresearch/sam2#download-checkpoints

Salvar em um local conhecido, por exemplo:
- Windows: `D:/SAM2/checkpoints/sam2.1_hiera_large.pt`
- Linux: `~/models/sam2.1_hiera_large.pt`

## Passo 6: Instalar Dependências do Projeto

```bash
cd sam-mosaic
pip install -e .
pip install rasterio fiona geopandas shapely scipy tqdm pyyaml
```

Para exportar shapefiles, instalar fiona via conda (mais confiável):

```bash
conda install -c conda-forge fiona -y
```

## Passo 7: Criar Pasta do Experimento

```bash
mkdir -p experiments/exp_full/output
```

## Passo 8: Criar Arquivo de Configuração

Criar o arquivo `experiments/exp_full/config.yaml` com o seguinte conteúdo:

```yaml
# Experiment: Full image segmentation
# Ajustar o caminho do checkpoint conforme seu sistema

sam:
  checkpoint: "CAMINHO/PARA/sam2.1_hiera_large.pt"  # <-- AJUSTAR
  model_type: "large"
  device: "cuda"

tiling:
  tile_size: 2000
  overlap: 0

base_tiles:
  grid:
    points_per_side: 64
  cascade:
    n_passes: 20
    thresholds:
      iou: [0.94, 0.75]
      stability: [0.97, 0.78]
    points_per_pass: 64
    point_erosion: 5

border_correction:
  zone_width: 100
  v_tiles:
    grid:
      n_across: 5
      n_along: 100
    cascade:
      n_passes: 5
      thresholds:
        iou: [0.90, 0.86]
        stability: [0.94, 0.90]
      points_per_pass: 64
      point_erosion: 5
  h_tiles:
    grid:
      n_across: 5
      n_along: 100
    cascade:
      n_passes: 5
      thresholds:
        iou: [0.90, 0.86]
        stability: [0.94, 0.90]
      points_per_pass: 64
      point_erosion: 5
  corner_tiles:
    grid:
      n_x: 5
      n_y: 5
    cascade:
      n_passes: 5
      thresholds:
        iou: [0.90, 0.86]
        stability: [0.94, 0.90]
      points_per_pass: 25
      point_erosion: 5

merge:
  priority: [2, 1, 0]

output:
  save_intermediate: true
  formats: ["tif", "shp"]
```

**IMPORTANTE:** Substituir `CAMINHO/PARA/sam2.1_hiera_large.pt` pelo caminho real do checkpoint.

## Passo 9: Criar Script de Execução

Criar o arquivo `experiments/exp_full/run.py`:

```python
#!/usr/bin/env python3
"""Run full image segmentation experiment."""

import os
os.environ["OMP_NUM_THREADS"] = "4"

import warnings
warnings.filterwarnings("ignore")

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import rasterio

# ============================================================
# CONFIGURAR ESTES CAMINHOS
# ============================================================
SOURCE_IMAGE = Path("CAMINHO/PARA/SUA/IMAGEM.tif")  # <-- AJUSTAR
# ============================================================

EXP_DIR = Path(__file__).parent
OUTPUT_DIR = EXP_DIR / "output"
CONFIG_FILE = EXP_DIR / "config.yaml"


def main():
    from sam_mosaic import load_config, Pipeline

    if not SOURCE_IMAGE.exists():
        print(f"ERRO: Imagem não encontrada: {SOURCE_IMAGE}")
        print("Ajuste o caminho SOURCE_IMAGE no script.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Imagem: {SOURCE_IMAGE}")
    print(f"Config: {CONFIG_FILE}")

    config = load_config(CONFIG_FILE)

    with rasterio.open(SOURCE_IMAGE) as src:
        width, height = src.width, src.height
        tile_size = config.tiling.tile_size
        n_cols = (width + tile_size - 1) // tile_size
        n_rows = (height + tile_size - 1) // tile_size

    print("\n" + "=" * 60)
    print("EXPERIMENTO SAM-MOSAIC")
    print("=" * 60)
    print(f"Imagem: {width} x {height} pixels ({width * height:,} total)")
    print(f"Tile size: {tile_size} x {tile_size}")
    print(f"Grid: {n_cols} x {n_rows} = {n_cols * n_rows} tiles")
    print(f"Descontinuidades: {n_cols-1} V, {n_rows-1} H, {(n_cols-1)*(n_rows-1)} corners")
    print("=" * 60 + "\n")

    start = time.time()
    pipeline = Pipeline(config)
    result = pipeline.run(SOURCE_IMAGE, OUTPUT_DIR)
    elapsed = time.time() - start

    print(f"\nCompleto em {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Output: {result}")

    # Estatísticas
    with rasterio.open(result) as src:
        total = 0
        covered = 0
        labels_set = set()

        for ji, window in src.block_windows(1):
            data = src.read(1, window=window)
            total += data.size
            covered += (data > 0).sum()
            labels_set.update(np.unique(data))

        labels_set.discard(0)

        print(f"\n{'=' * 60}")
        print("RESULTADOS")
        print("=" * 60)
        print(f"Cobertura: {covered/total*100:.2f}%")
        print(f"Instâncias: {len(labels_set):,}")
        print("=" * 60)


if __name__ == "__main__":
    main()
```

**IMPORTANTE:** Substituir `CAMINHO/PARA/SUA/IMAGEM.tif` pelo caminho real da imagem.

## Passo 10: Executar

```bash
conda activate sam_mosaic
cd sam-mosaic/experiments/exp_full
python run.py
```

## Outputs Esperados

Após a execução, a pasta `experiments/exp_full/output/` conterá:

| Arquivo | Descrição |
|---------|-----------|
| `band0_base.tif` | Segmentação dos tiles base |
| `band1_vh.tif` | Correção de bordas V/H |
| `band2_corners.tif` | Correção de corners |
| `merged_labels.tif` | Resultado final (raster) |
| `segments.shp` | Resultado final (shapefile) |

## Resultados Esperados

Para uma imagem de 15000x30000 pixels:

- **Tempo:** ~40-50 minutos (GPU RTX 3080 ou similar)
- **Cobertura:** ~95%
- **Instâncias:** ~19.000-20.000 polígonos

## Troubleshooting

### Erro: CUDA out of memory
- Reduzir `tile_size` para 1500 ou 1000
- Reduzir `points_per_side` para 32

### Erro: Module not found
```bash
pip install -e .
```

### Erro ao exportar shapefile
```bash
conda install -c conda-forge fiona -y
```

### Warnings do GDAL
São apenas avisos, podem ser ignorados:
```
Warning 3: Cannot find gdalvrt.xsd (GDAL_DATA is not defined)
```

## Parâmetros Importantes

| Parâmetro | Valor Padrão | Descrição |
|-----------|--------------|-----------|
| `tile_size` | 2000 | Tamanho do tile em pixels |
| `points_per_side` | 64 | Grid de pontos (64x64=4096) |
| `n_passes` | 20 | Iterações do cascade (base) |
| `iou` | [0.94, 0.75] | Threshold IoU [início, fim] |
| `stability` | [0.97, 0.78] | Threshold estabilidade |

## Estrutura do Projeto

```
sam-mosaic/
├── src/sam_mosaic/       # Código fonte
├── scripts/              # Scripts utilitários
├── configs/              # Configs de exemplo
├── experiments/          # Seus experimentos (não versionado)
│   └── exp_full/
│       ├── config.yaml
│       ├── run.py
│       └── output/
└── README.md
```
