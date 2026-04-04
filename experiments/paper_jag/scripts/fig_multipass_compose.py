"""Multi-pass progression figure: 3 datasets x 4 frames + coverage curves."""
import numpy as np
from pathlib import Path
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker

out_base = Path(__file__).resolve().parents[1] / "output" / "multipass_progression"
fig_dir = Path(__file__).resolve().parents[1] / "submission" / "latex" / "figures"


def load_clean(path):
    """Load snapshot PNG, crop off matplotlib title/whitespace."""
    arr = np.array(Image.open(path))[:, :, :3]
    gray = arr.mean(axis=2)
    for r0 in range(arr.shape[0]):
        if (gray[r0] < 200).sum() / gray.shape[1] > 0.8:
            break
    for r1 in range(arr.shape[0] - 1, 0, -1):
        if (gray[r1] < 200).sum() / gray.shape[1] > 0.5:
            break
    for c0 in range(arr.shape[1]):
        if (gray[:, c0] < 200).sum() / gray.shape[0] > 0.5:
            break
    for c1 in range(arr.shape[1] - 1, 0, -1):
        if (gray[:, c1] < 200).sum() / gray.shape[0] > 0.5:
            break
    return arr[r0:r1 + 1, c0:c1 + 1]


# --- Coverage data ---
bsb1_cov = [
    31.8, 35.8, 36.9, 37.1, 37.1, 39.8, 40.0, 40.0, 40.0, 43.9,
    47.6, 47.8, 47.8, 50.7, 51.5, 52.7, 52.7, 54.3, 55.4, 57.2,
    59.8, 59.9, 62.1, 62.7, 63.7, 63.9, 65.7, 65.8, 66.2, 66.8,
    67.2, 68.2, 68.2, 68.6, 68.7, 69.4, 69.4, 70.3, 70.7, 70.8,
    70.8, 70.8, 72.4, 72.7, 72.7, 72.7, 73.0, 75.4, 76.0, 76.0,
    76.3, 76.6, 76.7, 76.7, 78.0, 82.7, 87.6, 87.7, 87.7, 87.7,
    87.7, 87.9, 88.0, 88.0, 88.0, 88.1, 88.1, 88.1, 88.1, 88.1,
    88.1, 88.2, 88.3, 88.3, 88.3, 88.3, 88.3, 88.5, 88.5, 88.5,
]

potsdam_cov = [
    42.4, 42.5, 43.3, 43.3, 47.3, 47.3, 47.3, 48.0, 48.1, 48.1,
    48.6, 48.6, 48.6, 48.7, 50.1, 52.5, 52.5, 52.5, 52.5, 52.9,
    69.2, 69.0, 69.2, 69.2, 69.2, 69.2, 69.2, 69.2, 69.2, 69.2,
    69.2, 69.4, 69.4, 69.5, 98.0, 98.0, 98.0, 98.1, 98.1, 98.1,
    98.1, 98.1, 98.1, 98.1, 98.2, 98.2, 98.2, 98.2,
]

plant23_cov = [
    56.9, 58.6, 68.1, 77.6, 79.9, 80.2, 80.3, 80.8, 84.2, 80.8,
    84.3, 84.3, 84.3, 84.3, 85.9, 96.9, 96.9, 96.9, 96.9, 96.9,
    97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0,
    97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0,
    97.0, 97.0, 97.0, 97.1, 97.1,
]

# Colors
C_BSB = '#c0392b'
C_POT = '#2980b9'
C_PLT = '#27ae60'

# --- Frame definitions ---
# (directory, filename, title_text)
rows = [
    {
        'dir': 'bsb1_residential',
        'color': C_BSB,
        'label': 'BSB-1\n(24 cm)',
        'frames': [
            ('pass_00_original.png', 'Original'),
            ('pass_01.png', 'Pass 1 (32%)'),
            ('pass_30.png', 'Pass 30 (67%)'),
            ('pass_57.png', 'Pass 57 (88%)'),
        ],
        'markers': [(0, 31.8), (29, 66.8), (56, 87.6)],
    },
    {
        'dir': 'potsdam1_urban',
        'color': C_POT,
        'label': 'Potsdam-1\n(5 cm)',
        'frames': [
            ('pass_00_original.png', 'Original'),
            ('pass_01.png', 'Pass 1 (42%)'),
            ('pass_10.png', 'Pass 10 (49%)'),
            ('pass_48.png', 'Pass 48 (98%)'),
        ],
        'markers': [(0, 42.4), (9, 48.1), (47, 98.2)],
    },
    {
        'dir': 'plant23_fields',
        'color': C_PLT,
        'label': 'Plant23\n(4.78 m)',
        'frames': [
            ('pass_00_original.png', 'Original'),
            ('pass_01.png', 'Pass 1 (57%)'),
            ('pass_09.png', 'Pass 9 (81%)'),
            ('pass_45.png', 'Pass 45 (97%)'),
        ],
        'markers': [(0, 56.9), (8, 84.2), (44, 97.1)],
    },
]

# --- Build figure ---
fig = plt.figure(figsize=(14, 15.5))
gs = GridSpec(4, 4, figure=fig, height_ratios=[1, 1, 1, 0.7],
             hspace=0.15, wspace=0.04,
             left=0.06, right=0.97, top=0.97, bottom=0.04)

letters = iter('abcdefghijklmn')

for row_idx, row_data in enumerate(rows):
    snap_dir = out_base / row_data['dir']
    for col_idx, (fname, title) in enumerate(row_data['frames']):
        ax = fig.add_subplot(gs[row_idx, col_idx])
        img = load_clean(snap_dir / fname)
        ax.imshow(img)
        ax.axis('off')

        letter = next(letters)
        # Title on top
        if row_idx == 0:
            ax.set_title(title, fontsize=10, pad=5, fontweight='bold')
        else:
            ax.set_title(title, fontsize=10, pad=5, fontweight='bold')

        # Panel letter
        ax.text(0.03, 0.97, f"({letter})", transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top', color='white',
                bbox=dict(facecolor='black', alpha=0.8,
                          boxstyle='round,pad=0.12', edgecolor='none'))

        # Row label on left column
        if col_idx == 0:
            ax.text(-0.08, 0.5, row_data['label'], transform=ax.transAxes,
                    fontsize=11, fontweight='bold', va='center', ha='center',
                    rotation=90, color=row_data['color'])

# --- Coverage curves ---
ax_c = fig.add_subplot(gs[3, :])

ax_c.plot(range(len(bsb1_cov)), bsb1_cov, '-', color=C_BSB, lw=2.5,
          label='BSB-1 (24 cm, urban)', zorder=3)
ax_c.plot(range(len(potsdam_cov)), potsdam_cov, '-', color=C_POT, lw=2.5,
          label='Potsdam-1 (5 cm, urban)', zorder=3)
ax_c.plot(range(len(plant23_cov)), plant23_cov, '-', color=C_PLT, lw=2.5,
          label='Plant23 (4.78 m, agricultural)', zorder=3)

# Markers for selected frames (skip Original = column 0)
for row_data in rows:
    for px, py in row_data['markers']:
        ax_c.plot(px, py, 'o', color=row_data['color'], ms=9, zorder=5,
                  markeredgecolor='white', markeredgewidth=1.5)

# Annotations
ax_c.annotate('threshold decay\nunlocks road/pavement',
              xy=(34, 98.0), xytext=(50, 80),
              fontsize=9, fontstyle='italic', color='#666666',
              arrowprops=dict(arrowstyle='->', color='#999999', lw=1.2),
              ha='center')

ax_c.annotate('gradual coverage gain\nover 80 passes',
              xy=(55, 82.7), xytext=(68, 62),
              fontsize=9, fontstyle='italic', color='#666666',
              arrowprops=dict(arrowstyle='->', color='#999999', lw=1.2),
              ha='center')

# Single-pass baseline
ax_c.axhline(y=30, color='#bdc3c7', ls='--', lw=1, zorder=1)
ax_c.text(1, 31.5, 'single-pass ($\\tau$=0.93)', fontsize=8,
          color='#aaaaaa', fontstyle='italic')

ax_c.set_xlabel('Pass number', fontsize=12, labelpad=5)
ax_c.set_ylabel('Coverage (%)', fontsize=12, labelpad=5)
ax_c.set_xlim(-1, 82)
ax_c.set_ylim(25, 102)
ax_c.legend(fontsize=10, loc='center right', framealpha=0.95,
            edgecolor='#dddddd', fancybox=True)
ax_c.grid(True, alpha=0.15, color='#cccccc')
ax_c.tick_params(labelsize=10)
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)

out_path = fig_dir / "fig3_multipass_progression.png"
plt.savefig(str(out_path), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out_path}")
