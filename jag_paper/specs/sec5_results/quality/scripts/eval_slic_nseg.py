"""Compute n̄ (mean segments per object) for SLIC baseline."""
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import rasterize
from skimage.segmentation import slic
from scipy import ndimage
from collections import defaultdict
import time

def p(msg):
    print(msg, flush=True)

def eval_per_object_fast(labels, gt_instances, instance_class, n_instances, H, W, min_frac=0.20):
    seg_areas = np.bincount(labels.ravel(), minlength=int(labels.max()) + 1)
    inst_areas = np.bincount(gt_instances.ravel(), minlength=n_instances + 1)

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

        # Bbox from instance raster
        mask = gt_instances == iid
        rows_idx, cols_idx = np.where(mask)
        pad = max(5, int(np.sqrt(area) * 0.2))
        r0, r1 = max(0, rows_idx.min() - pad), min(H, rows_idx.max() + 1 + pad)
        c0, c1 = max(0, cols_idx.min() - pad), min(W, cols_idx.max() + 1 + pad)

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


# ============================================================
# BSB-1
# ============================================================
p("=== BSB-1 (SLIC n=27000) ===")
t0 = time.time()

with rasterio.open('experiments/paper_jag/data/brasilia/bsb_1.tif') as src:
    img = src.read([1, 2, 3]).transpose(1, 2, 0)
    transform = src.transform
H, W = 8000, 8000

p("  SLIC...")
labels = slic(img, n_segments=27000, compactness=10, start_label=1)
p(f"  {len(np.unique(labels))} segments [{time.time()-t0:.0f}s]")

gdf = gpd.read_file('experiments/paper_jag/data/brasilia/BSB/True_Bsb.shp')
class_map = {'Edificação': 1, 'Arvore': 2, 'Carros': 3, 'Piscina': 4,
             'Vias': 5, 'Quadra_Esportes': 6, 'Lago': 7, 'Deck': 8}

shapes = []
instance_class = {}
iid = 1
for cn, cid in class_map.items():
    for _, row in gdf[gdf['NomeClasse'] == cn].iterrows():
        shapes.append((row.geometry, iid))
        instance_class[iid] = cn
        iid += 1

gt_instances = rasterize(shapes, out_shape=(H, W), transform=transform, fill=0, dtype=np.int32)
p(f"  {iid-1} instances [{time.time()-t0:.0f}s]")

results = eval_per_object_fast(labels, gt_instances, instance_class, iid-1, H, W)
p(f"\n  {'Class':20s} {'n':>5} {'Det%':>6} {'IoU':>6} {'n_seg':>6}")
p("  " + "-" * 50)
for cn in class_map.keys():
    r = results[cn]
    if r['n'] == 0: continue
    det = r['detected'] / r['n'] * 100
    iou = np.mean(r['iou_list']) if r['iou_list'] else 0
    nseg = np.mean(r['nseg_list']) if r['nseg_list'] else 0
    p(f"  {cn:20s} {r['n']:5d} {det:5.1f}% {iou:6.3f} {nseg:6.1f}")

# ============================================================
# Potsdam
# ============================================================
p(f"\n=== Potsdam-1 (SLIC n=2400) === [{time.time()-t0:.0f}s]")

with rasterio.open('experiments/paper_jag/data/potsdam/top_potsdam_3_13_RGB.tif') as src:
    img = src.read([1, 2, 3]).transpose(1, 2, 0)
H, W = 6000, 6000

p("  SLIC...")
labels = slic(img, n_segments=2400, compactness=10, start_label=1)
p(f"  {len(np.unique(labels))} segments")

with rasterio.open('/mnt/d/DATA/Potsdam/Potsdam/5_Labels_all/top_potsdam_3_13_label.tif') as src:
    gt_rgb = src.read([1, 2, 3]).transpose(1, 2, 0)

color_to_class = {(255,255,255):1, (0,0,255):2, (0,255,255):3, (0,255,0):4, (255,255,0):5, (255,0,0):6}
class_names = {1:'Impervious', 2:'Building', 3:'Low_veg', 4:'Tree', 5:'Car', 6:'Clutter'}

gt_class = np.zeros((H,W), dtype=np.uint8)
for (r,g,b), cid in color_to_class.items():
    gt_class[(gt_rgb[:,:,0]==r)&(gt_rgb[:,:,1]==g)&(gt_rgb[:,:,2]==b)] = cid

structure = ndimage.generate_binary_structure(2, 1)
gt_instances = np.zeros((H,W), dtype=np.int32)
instance_class = {}
iid = 1
for cid, cname in class_names.items():
    labeled, n = ndimage.label(gt_class == cid, structure=structure)
    for comp in range(1, n+1):
        cm = labeled == comp
        if cm.sum() < 10: continue
        gt_instances[cm] = iid
        instance_class[iid] = cname
        iid += 1
p(f"  {iid-1} instances")

results = eval_per_object_fast(labels, gt_instances, instance_class, iid-1, H, W)
p(f"\n  {'Class':20s} {'n':>5} {'Det%':>6} {'IoU':>6} {'n_seg':>6}")
p("  " + "-" * 50)
for cname in class_names.values():
    r = results[cname]
    if r['n'] == 0: continue
    det = r['detected'] / r['n'] * 100
    iou = np.mean(r['iou_list']) if r['iou_list'] else 0
    nseg = np.mean(r['nseg_list']) if r['nseg_list'] else 0
    p(f"  {cname:20s} {r['n']:5d} {det:5.1f}% {iou:6.3f} {nseg:6.1f}")

# ============================================================
# Plant23
# ============================================================
p(f"\n=== Plant23 (SLIC n=3500) === [{time.time()-t0:.0f}s]")

with rasterio.open('experiments/paper_jag/data/plant23/plant23_10k.tif') as src:
    img = src.read([1, 2, 3]).transpose(1, 2, 0)
    transform = src.transform
H, W = 10000, 10000
if img.dtype != np.uint8:
    if img.max() > 0:
        img = (img.astype(np.float32) / img.max() * 255).astype(np.uint8)

p("  SLIC...")
labels = slic(img, n_segments=3500, compactness=10, start_label=1)
p(f"  {len(np.unique(labels))} segments")

gdf = gpd.read_file('jag_paper/data/True/Verdade_Planet.shp')
class_names_p = {1: 'Pivot', 2: 'Crops', 3: 'Lakes'}
shapes = []
instance_class = {}
iid = 1
for cid in [1, 2, 3]:
    for _, row in gdf[gdf['Class'] == cid].iterrows():
        shapes.append((row.geometry, iid))
        instance_class[iid] = class_names_p[cid]
        iid += 1

gt_instances = rasterize(shapes, out_shape=(H, W), transform=transform, fill=0, dtype=np.int32)
p(f"  {iid-1} instances")

results = eval_per_object_fast(labels, gt_instances, instance_class, iid-1, H, W)
p(f"\n  {'Class':20s} {'n':>5} {'Det%':>6} {'IoU':>6} {'n_seg':>6}")
p("  " + "-" * 50)
for cname in ['Pivot', 'Crops', 'Lakes']:
    r = results[cname]
    if r['n'] == 0: continue
    det = r['detected'] / r['n'] * 100
    iou = np.mean(r['iou_list']) if r['iou_list'] else 0
    nseg = np.mean(r['nseg_list']) if r['nseg_list'] else 0
    p(f"  {cname:20s} {r['n']:5d} {det:5.1f}% {iou:6.3f} {nseg:6.1f}")

p(f"\nAll done! [{time.time()-t0:.0f}s]")
