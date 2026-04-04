"""Felzenszwalb evaluation — optimized: rasterize ALL instances at once."""
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import rasterize
from skimage.segmentation import felzenszwalb
from scipy import ndimage
from collections import defaultdict
import time

def p(msg):
    print(msg, flush=True)


def compute_asa_with_other(labels, gt_class, other_id, n_classes):
    """ASA with background as 'Other' class."""
    gt_full = gt_class.copy()
    gt_full[gt_class == 0] = other_id
    gt_mask = gt_full > 0  # everything
    total = gt_mask.sum()

    lab_flat = labels.ravel()
    cls_flat = gt_full.ravel()

    unique_labels, inverse = np.unique(lab_flat, return_inverse=True)
    n_unique = len(unique_labels)
    combined = inverse.astype(np.int64) * n_classes + cls_flat.astype(np.int64)
    counts = np.bincount(combined, minlength=n_unique * n_classes).reshape(n_unique, n_classes)

    seg_majority_idx = counts[:, 1:].argmax(axis=1) + 1
    majority_counts = counts[:, 1:].max(axis=1)
    global_asa = majority_counts.sum() / total

    max_label = int(labels.max())
    seg_majority_map = np.zeros(max_label + 1, dtype=np.int32)
    seg_majority_map[unique_labels] = seg_majority_idx

    per_class = {}
    for cid in range(1, n_classes):
        cpx = (gt_full == cid).sum()
        if cpx == 0:
            continue
        lab_c = labels[gt_full == cid]
        pm = seg_majority_map[lab_c]
        per_class[cid] = (pm == cid).sum() / cpx

    return global_asa, per_class


def compute_asa_no_other(labels, gt_class, n_classes):
    """ASA without Other — Potsdam style (100% GT coverage)."""
    gt_mask = gt_class > 0
    total = gt_mask.sum()
    if total == 0:
        return 0.0, {}

    lab_flat = labels.ravel()[gt_mask.ravel()]
    cls_flat = gt_class.ravel()[gt_mask.ravel()]

    unique_labels, inverse = np.unique(lab_flat, return_inverse=True)
    n_unique = len(unique_labels)
    combined = inverse.astype(np.int64) * n_classes + cls_flat.astype(np.int64)
    counts = np.bincount(combined, minlength=n_unique * n_classes).reshape(n_unique, n_classes)

    seg_majority_idx = counts[:, 1:].argmax(axis=1) + 1
    majority_counts = counts[:, 1:].max(axis=1)
    global_asa = majority_counts.sum() / total

    max_label = int(labels.max())
    seg_majority_map = np.zeros(max_label + 1, dtype=np.int32)
    seg_majority_map[unique_labels] = seg_majority_idx

    per_class = {}
    for cid in range(1, n_classes):
        cpx = (gt_class == cid).sum()
        if cpx == 0:
            continue
        lab_c = labels[gt_class == cid]
        pm = seg_majority_map[lab_c]
        per_class[cid] = (pm == cid).sum() / cpx

    return global_asa, per_class


def eval_per_object_fast(labels, gt_instances, instance_class, n_instances, H, W, min_frac=0.20):
    """Fast per-object eval using precomputed instance raster."""
    seg_areas = np.bincount(labels.ravel(), minlength=int(labels.max()) + 1)

    # Precompute instance bboxes and areas from the raster
    p("    Computing instance properties...")
    inst_areas = np.bincount(gt_instances.ravel(), minlength=n_instances + 1)

    # Get bboxes efficiently
    rows, cols = np.indices((H, W))
    inst_row_min = np.full(n_instances + 1, H, dtype=np.int32)
    inst_row_max = np.zeros(n_instances + 1, dtype=np.int32)
    inst_col_min = np.full(n_instances + 1, W, dtype=np.int32)
    inst_col_max = np.zeros(n_instances + 1, dtype=np.int32)

    for iid in range(1, n_instances + 1):
        if inst_areas[iid] < 10:
            continue
        mask = gt_instances == iid
        r_idx = rows[mask]
        c_idx = cols[mask]
        inst_row_min[iid] = r_idx.min()
        inst_row_max[iid] = r_idx.max() + 1
        inst_col_min[iid] = c_idx.min()
        inst_col_max[iid] = c_idx.max() + 1

    p("    Evaluating instances...")
    class_results = defaultdict(lambda: {
        'n': 0, 'detected': 0, 'giant': 0, 'missed': 0,
        'iou_list': [], 'nseg_list': []
    })

    for iid in range(1, n_instances + 1):
        area = inst_areas[iid]
        if area < 10:
            continue
        cname = instance_class.get(iid)
        if cname is None:
            continue
        class_results[cname]['n'] += 1

        pad = max(5, int(np.sqrt(area) * 0.2))
        r0 = max(0, inst_row_min[iid] - pad)
        r1 = min(H, inst_row_max[iid] + pad)
        c0 = max(0, inst_col_min[iid] - pad)
        c1 = min(W, inst_col_max[iid] + pad)

        gt_crop = gt_instances[r0:r1, c0:c1] == iid
        lab_crop = labels[r0:r1, c0:c1]

        overlapping = set(np.unique(lab_crop[gt_crop])) - {0}
        if not overlapping:
            class_results[cname]['missed'] += 1
            continue

        good = []
        for seg_id in overlapping:
            seg_mask = lab_crop == seg_id
            inside = (seg_mask & gt_crop).sum()
            seg_total = seg_areas[seg_id]
            if seg_total > 0 and inside / seg_total > min_frac:
                good.append(seg_id)

        if not good:
            class_results[cname]['giant'] += 1
            continue

        class_results[cname]['detected'] += 1
        merged = np.zeros_like(gt_crop, dtype=bool)
        for seg_id in good:
            merged |= (lab_crop == seg_id)
        intersection = (merged & gt_crop).sum()
        union = (merged | gt_crop).sum()
        iou = intersection / union if union > 0 else 0
        class_results[cname]['iou_list'].append(iou)
        class_results[cname]['nseg_list'].append(len(good))

    return class_results


def print_results(class_results, class_order):
    p(f"  {'Class':20s} {'n':>5} {'Det%':>6} {'Giant':>5} {'Miss':>5} {'IoU':>6} {'n_seg':>6}")
    p("  " + "-" * 55)
    for cname in class_order:
        r = class_results[cname]
        if r['n'] == 0:
            continue
        det = r['detected'] / r['n'] * 100
        iou = np.mean(r['iou_list']) if r['iou_list'] else 0
        nseg = np.mean(r['nseg_list']) if r['nseg_list'] else 0
        p(f"  {cname:20s} {r['n']:5d} {det:5.1f}% {r['giant']:5d} {r['missed']:5d} {iou:6.3f} {nseg:6.1f}")


# ============================================================
# BSB-1
# ============================================================
def eval_bsb1():
    p("\n" + "=" * 60)
    p("  BSB-1 — Felzenszwalb (scale=1569)")
    p("=" * 60)

    t0 = time.time()
    with rasterio.open('experiments/paper_jag/data/brasilia/bsb_1.tif') as src:
        img = src.read([1, 2, 3]).transpose(1, 2, 0)
        transform = src.transform
    H, W = 8000, 8000

    p(f"  Running Felzenszwalb...")
    labels = felzenszwalb(img, scale=1569, sigma=0.5, min_size=50) + 1
    p(f"  {len(np.unique(labels))} segments [{time.time()-t0:.0f}s]")

    # GT classes
    gdf = gpd.read_file('experiments/paper_jag/data/brasilia/BSB/True_Bsb.shp')
    class_map = {
        'Edificação': 1, 'Arvore': 2, 'Carros': 3, 'Piscina': 4,
        'Vias': 5, 'Quadra_Esportes': 6, 'Lago': 7, 'Deck': 8
    }
    class_names = {v: k for k, v in class_map.items()}
    class_names[9] = 'Other'

    gt_class = np.zeros((H, W), dtype=np.uint8)
    for cn, cid in class_map.items():
        subset = gdf[gdf['NomeClasse'] == cn]
        if len(subset) > 0:
            r = rasterize([(g, cid) for g in subset.geometry],
                          out_shape=(H, W), transform=transform, fill=0, dtype=np.uint8)
            gt_class = np.where(r > 0, r, gt_class)
    p(f"  GT rasterized [{time.time()-t0:.0f}s]")

    # ASA
    global_asa, per_class_asa = compute_asa_with_other(labels, gt_class, 9, 10)
    p(f"\n  ASA Global (with Other): {global_asa*100:.1f}%")
    for cid in sorted(per_class_asa.keys()):
        p(f"    {class_names.get(cid, f'C{cid}'):20s}: {per_class_asa[cid]*100:.1f}%")

    # Rasterize ALL instances at once (fast!)
    p(f"\n  Rasterizing instances...")
    shapes = []
    instance_class = {}
    iid = 1
    for cn, cid in class_map.items():
        subset = gdf[gdf['NomeClasse'] == cn]
        for _, row in subset.iterrows():
            shapes.append((row.geometry, iid))
            instance_class[iid] = cn
            iid += 1

    gt_instances = rasterize(shapes, out_shape=(H, W), transform=transform,
                             fill=0, dtype=np.int32)
    n_instances = iid - 1
    p(f"  {n_instances} instances rasterized [{time.time()-t0:.0f}s]")

    results = eval_per_object_fast(labels, gt_instances, instance_class, n_instances, H, W)
    p(f"\n  Per-object [{time.time()-t0:.0f}s]:")
    print_results(results, list(class_map.keys()))


# ============================================================
# Potsdam-1
# ============================================================
def eval_potsdam():
    p("\n" + "=" * 60)
    p("  Potsdam-1 — Felzenszwalb (scale=5739)")
    p("=" * 60)

    t0 = time.time()
    with rasterio.open('experiments/paper_jag/data/potsdam/top_potsdam_3_13_RGB.tif') as src:
        img = src.read([1, 2, 3]).transpose(1, 2, 0)
    H, W = 6000, 6000

    p(f"  Running Felzenszwalb...")
    labels = felzenszwalb(img, scale=5739, sigma=0.5, min_size=50) + 1
    p(f"  {len(np.unique(labels))} segments [{time.time()-t0:.0f}s]")

    # GT
    with rasterio.open('/mnt/d/DATA/Potsdam/Potsdam/5_Labels_all/top_potsdam_3_13_label.tif') as src:
        gt_rgb = src.read([1, 2, 3]).transpose(1, 2, 0)

    color_to_class = {
        (255, 255, 255): 1, (0, 0, 255): 2, (0, 255, 255): 3,
        (0, 255, 0): 4, (255, 255, 0): 5, (255, 0, 0): 6,
    }
    class_names = {1: 'Impervious', 2: 'Building', 3: 'Low_veg', 4: 'Tree', 5: 'Car', 6: 'Clutter'}

    gt_class = np.zeros((H, W), dtype=np.uint8)
    for (r, g, b), cid in color_to_class.items():
        mask = (gt_rgb[:,:,0] == r) & (gt_rgb[:,:,1] == g) & (gt_rgb[:,:,2] == b)
        gt_class[mask] = cid

    global_asa, per_class_asa = compute_asa_no_other(labels, gt_class, 7)
    p(f"\n  ASA Global: {global_asa*100:.1f}%")
    for cid in sorted(per_class_asa.keys()):
        p(f"    {class_names[cid]:20s}: {per_class_asa[cid]*100:.1f}%")

    # Per-object via connected components
    p(f"\n  Computing connected components...")
    structure = ndimage.generate_binary_structure(2, 1)
    gt_instances = np.zeros((H, W), dtype=np.int32)
    instance_class = {}
    iid = 1
    for cid, cname in class_names.items():
        labeled, n = ndimage.label(gt_class == cid, structure=structure)
        for comp in range(1, n + 1):
            comp_mask = labeled == comp
            if comp_mask.sum() < 10:
                continue
            gt_instances[comp_mask] = iid
            instance_class[iid] = cname
            iid += 1
    p(f"  {iid-1} instances [{time.time()-t0:.0f}s]")

    results = eval_per_object_fast(labels, gt_instances, instance_class, iid - 1, H, W)
    p(f"\n  Per-object [{time.time()-t0:.0f}s]:")
    print_results(results, list(class_names.values()))


# ============================================================
# Plant23
# ============================================================
def eval_plant23():
    p("\n" + "=" * 60)
    p("  Plant23 — Felzenszwalb (scale=5061)")
    p("=" * 60)

    t0 = time.time()
    with rasterio.open('experiments/paper_jag/data/plant23/plant23_10k.tif') as src:
        img = src.read([1, 2, 3]).transpose(1, 2, 0)
        transform = src.transform
    H, W = 10000, 10000

    if img.dtype != np.uint8:
        if img.max() > 0:
            img = (img.astype(np.float32) / img.max() * 255).astype(np.uint8)

    p(f"  Running Felzenszwalb...")
    labels = felzenszwalb(img, scale=5061, sigma=0.5, min_size=50) + 1
    p(f"  {len(np.unique(labels))} segments [{time.time()-t0:.0f}s]")

    # GT
    gdf = gpd.read_file('jag_paper/data/True/Verdade_Planet.shp')
    class_names = {1: 'Pivot', 2: 'Crops', 3: 'Lakes', 4: 'Other'}

    gt_class = np.zeros((H, W), dtype=np.uint8)
    for cid in [1, 2, 3]:
        subset = gdf[gdf['Class'] == cid]
        if len(subset) > 0:
            r = rasterize([(g, cid) for g in subset.geometry],
                          out_shape=(H, W), transform=transform, fill=0, dtype=np.uint8)
            gt_class = np.where(r > 0, r, gt_class)
    p(f"  GT rasterized [{time.time()-t0:.0f}s]")

    global_asa, per_class_asa = compute_asa_with_other(labels, gt_class, 4, 5)
    p(f"\n  ASA Global (with Other): {global_asa*100:.1f}%")
    for cid in sorted(per_class_asa.keys()):
        p(f"    {class_names.get(cid, f'C{cid}'):20s}: {per_class_asa[cid]*100:.1f}%")

    # Rasterize instances at once
    p(f"\n  Rasterizing instances...")
    shapes = []
    instance_class = {}
    iid = 1
    for cid in [1, 2, 3]:
        subset = gdf[gdf['Class'] == cid]
        for _, row in subset.iterrows():
            shapes.append((row.geometry, iid))
            instance_class[iid] = class_names[cid]
            iid += 1

    gt_instances = rasterize(shapes, out_shape=(H, W), transform=transform,
                             fill=0, dtype=np.int32)
    p(f"  {iid-1} instances [{time.time()-t0:.0f}s]")

    results = eval_per_object_fast(labels, gt_instances, instance_class, iid - 1, H, W)
    p(f"\n  Per-object [{time.time()-t0:.0f}s]:")
    print_results(results, ['Pivot', 'Crops', 'Lakes'])


if __name__ == '__main__':
    eval_bsb1()
    eval_potsdam()
    eval_plant23()
    p("\nAll done!")
