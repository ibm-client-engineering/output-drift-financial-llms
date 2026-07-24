#!/usr/bin/env python3
"""
Generate null-correlation scatter plot for JFDS paper.
Source of truth: verified unified JSON data matching Exhibit 2.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# --- Verified data from unified JSON files (validated against Exhibit 2) ---
verified = [
    ('Qwen2.5-7B',      'Compliance',  100, 38),
    ('Qwen2.5-7B',      'Portfolio',   100, 20),
    ('Qwen2.5-7B',      'DataOps',      94, 44),
    ('Granite-3.3',     'Compliance',  100, 34),
    ('Granite-3.3',     'Portfolio',    84, 54),
    ('Granite-3.3',     'DataOps',     100, 60),
    ('Claude Sonnet 4', 'Compliance',   82, 34),
    ('Claude Sonnet 4', 'Portfolio',    86, 68),
    ('Claude Sonnet 4', 'DataOps',      84, 10),
    ('Claude Opus 4', 'Compliance',   72, 64),
    ('Claude Opus 4', 'Portfolio',    78, 38),
    ('Gemini 2.0 Flash','Compliance',   68, 52),
    ('Gemini 2.0 Flash','Portfolio',    96, 62),
    ('Gemini 2.0 Flash','DataOps',      94, 34),
    # Claude Opus 4 DataOps: not in verified unified files, omitted
]

models = [x[0] for x in verified]
benchmarks = [x[1] for x in verified]
det = np.array([x[2] for x in verified], dtype=float)
acc = np.array([x[3] for x in verified], dtype=float)

# --- Correlation ---
r, p = stats.pearsonr(det, acc)
print(f"Pearson r = {r:.2f}, p = {p:.2f}, n = {len(det)}")

# --- Plot ---
fig, ax = plt.subplots(figsize=(6.5, 5))

bm_style = {
    'Compliance': ('o', '#1f77b4', 'Compliance Triage'),
    'Portfolio':  ('s', '#ff7f0e', 'Portfolio Constraint'),
    'DataOps':    ('^', '#2ca02c', 'DataOps Exception'),
}

for bm, (marker, color, label) in bm_style.items():
    mask = np.array([b == bm for b in benchmarks])
    if mask.any():
        ax.scatter(det[mask], acc[mask], marker=marker, s=70, alpha=0.85,
                   color=color, edgecolors='black', linewidth=0.5,
                   label=label, zorder=3)

# Label points
placed = []
for i, (x, y, model) in enumerate(zip(det, acc, models)):
    oy = 2.5
    for px, py in placed:
        if abs(x - px) < 8 and abs(y - py) < 5:
            oy = -4.0
            break
    ax.annotate(model, (x, y), fontsize=6.5, alpha=0.7,
                textcoords='offset points', xytext=(4, oy),
                ha='left', va='bottom' if oy > 0 else 'top')
    placed.append((x, y))

ax.set_xlabel('Decision Consistency (%)', fontsize=11)
ax.set_ylabel('Task Accuracy (%)', fontsize=11)
ax.set_xlim(63, 105)
ax.set_ylim(5, 75)

ax.text(0.03, 0.97,
        f'Pearson $r$ = {r:.2f} ($p$ = {p:.2f})\n$n$ = {len(det)} configurations',
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='gray', alpha=0.9))

ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()

out_dir = os.path.join(os.path.dirname(__file__), '..', 'figs')
os.makedirs(out_dir, exist_ok=True)
fig.savefig(os.path.join(out_dir, 'fig_null_correlation.pdf'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(out_dir, 'fig_null_correlation.png'), dpi=150, bbox_inches='tight')
print("Saved.")
