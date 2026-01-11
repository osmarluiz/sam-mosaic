# SAM2 Checkpoints

Download SAM2 checkpoints from the official repository:
https://github.com/facebookresearch/sam2

## Available Models

| Model | Size | Download |
|-------|------|----------|
| sam2.1_hiera_tiny | 149 MB | [Download](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt) |
| sam2.1_hiera_small | 176 MB | [Download](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt) |
| sam2.1_hiera_base_plus | 309 MB | [Download](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt) |
| sam2.1_hiera_large | 857 MB | [Download](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) |

## Quick Download

```bash
# Download the large model (recommended)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# Or using curl
curl -O https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

## Note

The `.pt` files are excluded from git via `.gitignore` due to their large size.
