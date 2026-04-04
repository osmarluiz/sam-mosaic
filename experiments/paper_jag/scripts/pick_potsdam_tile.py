"""Show all 36 Potsdam tiles as a grid to pick the best one."""
import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

img_path = Path(__file__).resolve().parents[1] / "data" / "potsdam" / "top_potsdam_3_13_RGB.tif"
out_path = Path(__file__).resolve().parents[1] / "output" / "multipass_progression" / "potsdam_tile_grid.png"

TILE_SIZE = 1000

with rasterio.open(img_path) as src:
    H, W = src.height, src.width
    full = src.read([1, 2, 3]).transpose(1, 2, 0)

n_rows = H // TILE_SIZE
n_cols = W // TILE_SIZE

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 18))
for r in range(n_rows):
    for c in range(n_cols):
        y0, x0 = r * TILE_SIZE, c * TILE_SIZE
        tile = full[y0:y0+TILE_SIZE, x0:x0+TILE_SIZE]
        ax = axes[r][c]
        ax.imshow(tile)
        ax.set_title(f"[{r},{c}]", fontsize=10, fontweight='bold')
        ax.axis('off')

plt.suptitle("Potsdam-1 tiles (6x6 grid, T=1000)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")
