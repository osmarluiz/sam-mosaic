"""Compute per-class ASA + global ASA + mIoU + global Detection for:
- RemoteSAMsing (best config per dataset)
- SLIC baseline
- Felzenszwalb baseline

Output: data for the unified quality tables in the paper.
"""
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import rasterize
from skimage.segmentation import slic, felzenszwalb
from scipy import ndimage

# ========================================================
# HELPERS
# ========================================================

def compute_asa_global_and_perclass(labels, gt_class, n_classes=9):
    """Compute global ASA and per-class ASA.

    Args:
        labels: (H, W) int array, 0=background
        gt_class: (H, W) uint8, 0=no GT, 1..N=classes
        n_classes: max class ID + 1

    Returns:
        global_asa: float
        per_class: dict {class_id: asa_float}
    """
    gt_mask = gt_class > 0
    total_gt = gt_mask.sum()
    if total_gt == 0:
        return 0.0, {}

    lab_flat = labels.ravel()[gt_mask.ravel()]
    cls_flat = gt_class.ravel()[gt_mask.ravel()]

    unique_labels, inverse = np.unique(lab_flat, return_inverse=True)
    n_unique = len(unique_labels)

    combined = inverse.astype(np.int64) * n_classes + cls_flat.astype(np.int64)
    counts = np.bincount(combined, minlength=n_unique * n_classes)
    counts = counts.reshape(n_unique, n_classes)

    # Majority class per segment (excluding bg=0)
    seg_majority_idx = counts[:, 1:].argmax(axis=1) + 1  # 1-based class
    majority_counts = counts[:, 1:].max(axis=1)

    # Global ASA
    global_asa = majority_counts.sum() / total_gt

    # Build lookup: original label -> majority class
    max_label = labels.max()
    seg_majority_map = np.zeros(max_label + 1, dtype=np.int32)
    seg_majority_map[unique_labels] = seg_majority_idx

    # Per-class ASA
    per_class = {}
    for cid in range(1, n_classes):
        class_pixels = gt_class == cid
        n_pixels = class_pixels.sum()
        if n_pixels == 0:
            continue
        lab_this = labels[class_pixels]
        # For unsegmented pixels (label=0), majority is 0 (wrong)
        pixel_majority = seg_majority_map[lab_this]
        correct = (pixel_majority == cid).sum()
        per_class[cid] = correct / n_pixels

    return global_asa, per_class


def compute_asa_with_other(labels, gt_class, n_classes_with_other):
    """ASA treating unannotated background as 'Other' class.

    Assign class_id = n_classes_with_other - 1 to all pixels where gt_class == 0.
    """
    gt_with_other = gt_class.copy()
    other_id = n_classes_with_other - 1
    gt_with_other[gt_class == 0] = other_id

    return compute_asa_global_and_perclass(labels, gt_with_other, n_classes_with_other)


# ========================================================
# DATASET 1: BSB-1
# ========================================================

def eval_bsb1():
    print("\n" + "=" * 60)
    print("  BSB-1 (24cm)")
    print("=" * 60)

    img_path = 'experiments/paper_jag/data/brasilia/bsb_1.tif'
    transform = rasterio.open(img_path).transform
    H, W = 8000, 8000

    # GT
    gdf = gpd.read_file('experiments/paper_jag/data/brasilia/BSB/True_Bsb.shp')
    class_map = {
        'Edificação': 1, 'Arvore': 2, 'Carros': 3, 'Piscina': 4,
        'Vias': 5, 'Quadra_Esportes': 6, 'Lago': 7, 'Deck': 8
    }
    class_names = {v: k for k, v in class_map.items()}
    class_names[9] = 'Other'

    gt_class = np.zeros((H, W), dtype=np.uint8)
    for class_name, class_id in class_map.items():
        subset = gdf[gdf['NomeClasse'] == class_name]
        if len(subset) == 0:
            continue
        r = rasterize([(geom, class_id) for geom in subset.geometry],
                      out_shape=(H, W), transform=transform, fill=0, dtype=np.uint8)
        gt_class = np.where(r > 0, r, gt_class)

    # Labels to evaluate
    label_configs = {
        'RS (Dense T=250)': 'experiments/paper_jag/output/v3/ablation_strategy_tilesize/bsb1_dense_250/labels.tif',
        'RS (Dense T=500)': 'experiments/paper_jag/output/v3/ablation_strategy_tilesize/bsb1_dense_500/labels.tif',
        'RS (Dense T=1000)': 'experiments/paper_jag/output/v3/ablation_strategy_tilesize/bsb1_dense_1000/labels.tif',
    }

    # SLIC baseline
    print("  Computing SLIC...", flush=True)
    with rasterio.open(img_path) as src:
        img = src.read([1, 2, 3]).transpose(1, 2, 0)
    slic_labels = slic(img, n_segments=27000, compactness=10, start_label=1)

    # Felzenszwalb baseline
    print("  Computing Felzenszwalb...", flush=True)
    felz_labels = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
    felz_labels += 1  # start from 1

    # Add baselines
    label_configs['SLIC'] = slic_labels
    label_configs['Felzenszwalb'] = felz_labels

    # Compute with Other class (class 9)
    n_classes_other = 10  # 0=unused, 1-8=classes, 9=Other
    print()
    for name, path_or_arr in label_configs.items():
        if isinstance(path_or_arr, str):
            labels = rasterio.open(path_or_arr).read(1)
        else:
            labels = path_or_arr

        global_asa, per_class = compute_asa_with_other(labels, gt_class, n_classes_other)
        print(f"  {name}:")
        print(f"    Global ASA (with Other): {global_asa * 100:.1f}%")
        for cid in sorted(per_class.keys()):
            cname = class_names.get(cid, f'Class{cid}')
            print(f"    {cname:20s}: {per_class[cid] * 100:.1f}%")
        print()


# ========================================================
# DATASET 2: Potsdam-1
# ========================================================

def eval_potsdam():
    print("\n" + "=" * 60)
    print("  Potsdam-1 (5cm)")
    print("=" * 60)

    img_path = 'experiments/paper_jag/data/potsdam/top_potsdam_3_13_RGB.tif'
    gt_path = '/mnt/d/DATA/Potsdam/Potsdam/5_Labels_all/top_potsdam_3_13_label.tif'
    H, W = 6000, 6000

    # GT: ISPRS RGB label image -> class IDs
    with rasterio.open(gt_path) as src:
        gt_rgb = src.read([1, 2, 3]).transpose(1, 2, 0)

    # ISPRS color map
    color_to_class = {
        (255, 255, 255): 1,  # Impervious
        (0, 0, 255): 2,      # Building
        (0, 255, 255): 3,    # Low vegetation
        (0, 255, 0): 4,      # Tree
        (255, 255, 0): 5,    # Car
        (255, 0, 0): 6,      # Clutter
    }
    class_names = {1: 'Impervious', 2: 'Building', 3: 'Low_veg', 4: 'Tree', 5: 'Car', 6: 'Clutter'}

    gt_class = np.zeros((H, W), dtype=np.uint8)
    for (r, g, b), cid in color_to_class.items():
        mask = (gt_rgb[:, :, 0] == r) & (gt_rgb[:, :, 1] == g) & (gt_rgb[:, :, 2] == b)
        gt_class[mask] = cid

    # Labels
    label_configs = {
        'RS (Dense T=1000)': 'experiments/paper_jag/output/v3/ablation_strategy_tilesize/potsdam1_dense_1000/labels.tif',
    }

    # SLIC
    print("  Computing SLIC...", flush=True)
    with rasterio.open(img_path) as src:
        img = src.read([1, 2, 3]).transpose(1, 2, 0)
    slic_labels = slic(img, n_segments=2400, compactness=10, start_label=1)

    # Felzenszwalb
    print("  Computing Felzenszwalb...", flush=True)
    felz_labels = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
    felz_labels += 1

    label_configs['SLIC'] = slic_labels
    label_configs['Felzenszwalb'] = felz_labels

    n_classes = 7  # 0=unused, 1-6=classes
    print()
    for name, path_or_arr in label_configs.items():
        if isinstance(path_or_arr, str):
            labels = rasterio.open(path_or_arr).read(1)
        else:
            labels = path_or_arr

        global_asa, per_class = compute_asa_global_and_perclass(labels, gt_class, n_classes)
        print(f"  {name}:")
        print(f"    Global ASA: {global_asa * 100:.1f}%")
        for cid in sorted(per_class.keys()):
            cname = class_names.get(cid, f'Class{cid}')
            print(f"    {cname:20s}: {per_class[cid] * 100:.1f}%")
        print()


# ========================================================
# DATASET 3: Plant23
# ========================================================

def eval_plant23():
    print("\n" + "=" * 60)
    print("  Plant23 (4.78m)")
    print("=" * 60)

    img_path = 'experiments/paper_jag/data/plant23/plant23_10k.tif'
    H, W = 10000, 10000
    transform = rasterio.open(img_path).transform

    # GT — column 'Class' with int values: 0=background, 1=Pivot, 2=Crops, 3=Lakes
    gdf = gpd.read_file('jag_paper/data/True/Verdade_Planet.shp')
    class_names = {1: 'Pivot', 2: 'Crops', 3: 'Lakes', 4: 'Other'}

    gt_class = np.zeros((H, W), dtype=np.uint8)
    for class_id in [1, 2, 3]:
        subset = gdf[gdf['Class'] == class_id]
        if len(subset) > 0:
            r = rasterize([(geom, class_id) for geom in subset.geometry],
                          out_shape=(H, W), transform=transform, fill=0, dtype=np.uint8)
            gt_class = np.where(r > 0, r, gt_class)

    # Labels
    label_configs = {
        'RS (Dense T=1000)': 'experiments/paper_jag/output/v3/ablation_strategy_tilesize/plant23_dense_1000/labels.tif',
    }

    # SLIC
    print("  Computing SLIC...", flush=True)
    with rasterio.open(img_path) as src:
        img = src.read([1, 2, 3]).transpose(1, 2, 0)
    if img.dtype != np.uint8:
        img = (img.astype(np.float32) / img.max() * 255).astype(np.uint8) if img.max() > 0 else img.astype(np.uint8)
    slic_labels = slic(img, n_segments=3500, compactness=10, start_label=1)

    # Felzenszwalb
    print("  Computing Felzenszwalb...", flush=True)
    felz_labels = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
    felz_labels += 1

    label_configs['SLIC'] = slic_labels
    label_configs['Felzenszwalb'] = felz_labels

    # With Other class (class 4)
    n_classes_other = 5  # 0=unused, 1-3=classes, 4=Other
    print()
    for name, path_or_arr in label_configs.items():
        if isinstance(path_or_arr, str):
            labels = rasterio.open(path_or_arr).read(1)
        else:
            labels = path_or_arr

        global_asa, per_class = compute_asa_with_other(labels, gt_class, n_classes_other)
        print(f"  {name}:")
        print(f"    Global ASA (with Other): {global_asa * 100:.1f}%")
        for cid in sorted(per_class.keys()):
            cname = class_names.get(cid, f'Class{cid}')
            print(f"    {cname:20s}: {per_class[cid] * 100:.1f}%")
        print()


# ========================================================
# MAIN
# ========================================================

if __name__ == '__main__':
    # eval_bsb1()    # DONE
    # eval_potsdam()  # DONE
    eval_plant23()
    print("\nAll done!")
