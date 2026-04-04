# §5 Results — Spec v2 (2026-04-04)

## Structure

```
§5 Results
  [Fig 2 hero + roadmap paragraph]
  §5.1 Pipeline Configuration Analysis
    §5.1.1 Component Contributions (Tab 3)
    §5.1.2 Point Strategy (Tab 4)
    §5.1.3 Tile Size (Tab 5, Fig 4)
  §5.2 Boundary Merge (Fig 5, Tab 6, Fig 6)
  §5.3 Segmentation Quality (Tab 7, Tab 8)
```

## §5 Opening

- Fig 2: Full segmentation hero (3 datasets)
- Roadmap paragraph: "The evaluation proceeds in three stages: pipeline configuration analysis (§5.1) identifies the optimal settings through component ablation, point strategy comparison, and tile size analysis; boundary merge validation (§5.2) confirms artifact-free tile stitching; and segmentation quality assessment (§5.3) evaluates the final output against ground truth and traditional baselines."

## §5.1 Pipeline Configuration Analysis

### §5.1.1 Component Contributions

**Tab 3:** 7 configs × 3 scenes × {Coverage%, Time(s)} + Avg ΔCov
- All configs use K-means T=1000 (fixed reference)
- Configs: full pipeline, single-pass τ=0.93/0.86/0.70, no black mask, fixed threshold τ=0.86, no padding

**Paragraph order:**
1. Overview + single-pass gap: "Full pipeline 89-98%. Single-pass at τ=0.93 covers only 30-68%. Even at τ=0.70, single-pass reaches only 77-93%."
2. Adaptive threshold (largest contribution): "+13.9pp avg. Fixed threshold at τ=0.86 cannot access high-quality masks at τ=0.93 nor additional coverage below τ=0.86."
3. Black mask (surprise): "+8.6pp avg. Without it, processing is 14% SLOWER (2614 vs 2293s BSB) AND covers LESS. Generates 27% more redundant segments (14,767 vs 11,624 BSB). SAM2 re-segments already-covered regions."
4. Padding + cost: "<1pp coverage, boundary quality in §5.2. Full pipeline 3-7× slower than single-pass."

### §5.1.2 Point Strategy

**Tab 4:** 7 scenes × {K-means Cov, Dense Cov, Δ, K-means Seg, Dense Seg}
- Uses full pipeline, T=1000

**Paragraphs:**
1. "Dense Grid +1-3pp on BSB urban, tied on Potsdam/Plant23. Produces 4-13% more segments."
2. "Dense Grid maintains uniform density on fragmented residual areas. K-means concentrates centroids in largest patches, potentially missing small fragments."
3. Transition: "Having established Dense Grid as preferred, the tile-size analysis adopts it."

### §5.1.3 Tile Size

**Tab 5:** 3 scenes × 3 sizes × {Coverage%, ASA%, Time(s)} — uses Dense Grid

**Fig 4:** Tile size × car detection (BSB-1)

**Paragraphs:**
1. "T=250 adds 1-7pp coverage, 9× slower processing."
2. "ASA degrades at small tiles on Potsdam (75.8→46.0%) and Plant23 (98.0→78.7%). At 5cm GSD, T=250 covers 12.5×12.5m → single-segment tiles → merge chaining."
3. "Fig 4: BSB-1 car detection — 91% detected at T=250 vs 40% at T=1000."
4. Recommended config: "Dense T=250 for BSB-1, Dense T=1000 for Potsdam/Plant23. Evaluated in Section 5.3."

NOTE: "Tile size as implicit scale parameter" interpretation → §6 Discussion.

## §5.2 Boundary Merge

**Fig 5:** Naive vs best-match merge (visual, 3 datasets)
**Tab 6:** 7 scenes × {Naive Seg/Merges, Best-match Seg/Merges, ΔSeg}
**Fig 6:** Padding effect (visual)

**Paragraphs:**
1. "Naive merge creates mega-segments (83% of crop in Potsdam example). Best-match requires each segment to commit to one partner."
2. "Tab 6: best-match produces 15-20% fewer merges, preserving 5-10% more segments."
3. "Fig 6: without padding, segments terminate at edges; with padding, continuous segmentation."
4. Bridge: "All quality results in Section 5.3 use best-match merge with contextual padding."

## §5.3 Segmentation Quality

**Tab 7:** RS per-object detailed — 3 datasets × all classes × {n, Det%, IoU, ASA%, n̄} + Global row (mDet, mIoU, ASA)
- BSB-1 at Dense T=250 (8 classes + global)
- Potsdam-1 at Dense T=1000 (6 classes + global)
- Plant23 at Dense T=1000 (3 classes + global)

**Tab 8:** Baseline comparison — 3 datasets × all classes × 3 methods × {Det%, IoU} + n̄ for RS + ASA global + segment counts
- SLIC: n_segments matched to RS, compactness=10
- Felzenszwalb: scale calibrated to closest achievable segment count, sigma=0.5, min_size=50
- Segment counts in caption

**Paragraphs:**
1. Intro + coverage gap: "Evaluates final output at recommended configs. RS covers 88-98%; uncovered 2-12% penalized as 'Other' in ASA and as missed in per-object. SLIC/Felz cover 100% by design."
2. Tab 7 overview
3. Defined objects: "Buildings, cars, fields, distinct trees → >81% Det, >0.79 IoU, n̄ close to 1.0. Buildings n̄=2.8 (roof+shadow). Tile size effect: BSB car Det 41%→81% from T=1000→T=250."
4. Challenging classes: "Roads 57.4% Det, n̄=30.1. Low veg 50.5%, impervious 56.3%. Lack distinct boundaries for SAM2."
5. Plant23: "Pivots 100% Det (IoU 0.99), crops 94.6% (IoU 0.98), despite MNF false-color."
6. Tab 8 intro: "Compares against SLIC and Felzenszwalb at approximately matched segment counts."
7. Baselines: "RS highest IoU on all 10 classes. SLIC 0% cars BSB, 3.5% Potsdam (uniform grid). Felz 44-86% Det but IoU 0.03-0.57 (poor boundary alignment). Potsdam ASA: SLIC 78.9 vs RS 75.8 (merge chaining), but RS wins every per-object metric."
8. Time: "SLIC 30-70s, Felz 150-600s, RS 1,400-5,700s at T=1000."
9. Factual summary (NO interpretation): "Across three datasets, RS achieves highest IoU on every class with competitive ASA."

NOTE: "Sensor agnosticism" and "OBIA suitability" interpretations → §6 Discussion.

## Data inventory

| Tab | Section | Data | Status |
|-----|---------|------|--------|
| 3 | §5.1.1 | Ablation coverage + time | ✅ Complete |
| 4 | §5.1.2 | Strategy comparison | ✅ Complete |
| 5 | §5.1.3 | Tile size cov + ASA + time | ✅ Complete |
| 6 | §5.2 | Merge statistics | ✅ Complete |
| 7 | §5.3 | RS per-object + ASA per class | ✅ Complete (need mDet/mIoU) |
| 8 | §5.3 | Baselines Det/IoU/n̄ + ASA | ✅ Complete |

| Fig | Section | Content | Status |
|-----|---------|---------|--------|
| 2 | §5 intro | Full segmentation hero | ✅ Exists |
| 4 | §5.1.3 | Tile size cars | ✅ Exists |
| 5 | §5.2 | Merge comparison | ✅ Exists |
| 6 | §5.2 | Padding effect | ✅ Exists |

## §4 adjustments needed
1. Reformulate strategy+tilesize experiment description to match what §5 shows
2. Remove Object Recall from metric definitions (report in text only)
3. Add/keep sensitivity analysis mention or remove from §4
4. Add sentence justifying SLIC/Felz as baselines (standard in OBIA literature)

## Cross-references
- §5.1.1 → §3.1 ("components described in Section 3.1")
- §5.1.2 → §5.1.1 ("using the full pipeline")
- §5.1.3 → §5.1.2 ("Having established Dense Grid")
- §5.1.3 → §5.3 ("evaluated in Section 5.3")
- §5.2 → §3.2 ("merge algorithm described in Section 3.2")
- §5.2 → §5.1.1 ("padding contributes <1pp, Table 3")
- §5.2 → §5.3 ("All quality results in Section 5.3 use best-match merge")
- §5.3 → §5.1 ("best configuration per dataset, Section 5.1")
