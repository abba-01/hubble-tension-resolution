#!/usr/bin/env python3
"""
TASK 6: Generate Publication Figures
Purpose: Create 6 publication-quality figures for the empirical validation
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Set base directory
BASE_DIR = Path("/run/media/root/OP01/got/hubble")
DATA_DIR = BASE_DIR / "data" / "vizier_data"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set publication style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.dpi'] = 150

print("=" * 80)
print("TASK 6: PUBLICATION FIGURES GENERATION")
print("=" * 80)
print()

# Load data
cov_matrix = np.load(RESULTS_DIR / "empirical_covariance_210x210.npy")
corr_matrix = np.load(RESULTS_DIR / "empirical_correlation_210x210.npy")
df = pd.read_csv(DATA_DIR / "J_ApJ_826_56_table3.csv")

with open(RESULTS_DIR / "covariance_eigenspectrum.json") as f:
    eigenspectrum = json.load(f)

with open(RESULTS_DIR / "anchor_tensors.json") as f:
    anchor_tensors = json.load(f)

with open(RESULTS_DIR / "concordance_empirical.json") as f:
    concordance = json.load(f)

with open(RESULTS_DIR / "comparison_phase_a_vs_c.json") as f:
    comparison = json.load(f)

print("[1/6] Generating Figure 1: Covariance Heatmap...")

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cov_matrix, cmap='viridis', aspect='auto')
ax.set_title('Empirical Systematic Covariance Matrix (Riess+ 2016)', fontsize=14, fontweight='bold')
ax.set_xlabel('Measurement Index')
ax.set_ylabel('Measurement Index')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Covariance [(km/s/Mpc)²]', rotation=270, labelpad=20)
plt.tight_layout()
fig1_path = FIGURES_DIR / "covariance_heatmap.pdf"
plt.savefig(fig1_path, bbox_inches='tight')
plt.close()
print(f"✅ Figure 1 saved: {fig1_path}")

print("[2/6] Generating Figure 2: Eigenspectrum...")

fig, ax = plt.subplots(figsize=(10, 6))
n_modes = min(10, len(eigenspectrum['eigenvalues']))
x = np.arange(1, n_modes + 1)
eigenvalues = eigenspectrum['eigenvalues'][:n_modes]
cumulative = eigenspectrum['cumulative_variance'][:n_modes]

ax.bar(x, eigenvalues, color='steelblue', alpha=0.7, label='Eigenvalues')
ax.set_xlabel('Mode Number')
ax.set_ylabel('Eigenvalue', color='steelblue')
ax.tick_params(axis='y', labelcolor='steelblue')

ax2 = ax.twinx()
ax2.plot(x, [c*100 for c in cumulative], 'ro-', linewidth=2, markersize=6, label='Cumulative Variance')
ax2.set_ylabel('Cumulative Variance (%)', color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.axhline(66.6, color='green', linestyle='--', alpha=0.5)
ax2.text(n_modes*0.7, 68, f"Top 3: {eigenspectrum['top3_variance_pct']:.1f}%", fontsize=10)

ax.set_title('Systematic Covariance Eigenspectrum', fontsize=14, fontweight='bold')
ax.set_xticks(x)
plt.tight_layout()
fig2_path = FIGURES_DIR / "eigenspectrum.pdf"
plt.savefig(fig2_path, bbox_inches='tight')
plt.close()
print(f"✅ Figure 2 saved: {fig2_path}")

print("[3/6] Generating Figure 3: Anchor Comparison...")

fig, ax = plt.subplots(figsize=(10, 6))

# Prepare data for each primary anchor
anchors = ['N', 'M', 'L']
anchor_names = {'N': 'NGC4258', 'M': 'Milky Way', 'L': 'LMC'}
positions = [1, 2, 3]

for i, anchor in enumerate(anchors):
    mask = df['Anc'] == anchor
    h0_vals = df[mask]['H0'].values

    # Box plot
    bp = ax.boxplot([h0_vals], positions=[positions[i]], widths=0.6,
                     patch_artist=True, showmeans=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_alpha(0.7)

# Planck reference line
ax.axhline(67.4, color='red', linestyle='--', linewidth=2, label='Planck 2018: 67.4 km/s/Mpc')

ax.set_xticks(positions)
ax.set_xticklabels([anchor_names[a] for a in anchors])
ax.set_ylabel('H₀ (km/s/Mpc)')
ax.set_title('Anchor-Dependent H₀ Measurements', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig3_path = FIGURES_DIR / "anchor_comparison.pdf"
plt.savefig(fig3_path, bbox_inches='tight')
plt.close()
print(f"✅ Figure 3 saved: {fig3_path}")

print("[4/6] Generating Figure 4: Concordance Intervals...")

fig, ax = plt.subplots(figsize=(12, 6))

# Planck (early)
planck_h0 = 67.4
planck_unc = 0.5
y_pos = 0

ax.errorbar([planck_h0], [y_pos], xerr=[planck_unc], fmt='o', color='blue',
            markersize=8, capsize=5, capthick=2, label='Planck (CMB)', linewidth=2)

# Late-time anchors
y_pos = 1
for anchor_code in ['N', 'M', 'L']:
    tensor = anchor_tensors[anchor_code]
    h0 = tensor['H0_mean']
    unc = tensor['H0_uncertainty']
    label = f"{tensor['anchor_name']} (SH0ES)"

    ax.errorbar([h0], [y_pos], xerr=[unc], fmt='s', markersize=8,
                capsize=5, capthick=2, label=label, linewidth=2)
    y_pos += 1

# Phase A merged (from comparison)
phase_a_gap = comparison['Phase_A']['gap_km_s_Mpc']
phase_a_center = (planck_h0 + 73.0) / 2  # Approximate
y_pos += 0.5
ax.plot([phase_a_center - 1, phase_a_center + 1], [y_pos, y_pos], 'g-', linewidth=3,
        label=f"Phase A: {phase_a_gap:.2f} km/s/Mpc gap")

# Phase C average
phase_c_gap = comparison['Phase_C']['average_gap_km_s_Mpc']
y_pos += 0.5
ax.plot([phase_a_center - 1.5, phase_a_center + 1.5], [y_pos, y_pos], 'm-', linewidth=3,
        label=f"Phase C: {phase_c_gap:.2f} km/s/Mpc gap (avg)")

ax.set_ylabel('Measurement')
ax.set_xlabel('H₀ (km/s/Mpc)')
ax.set_title('Hubble Tension Resolution: Published vs Empirical Covariance', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3, axis='x')
ax.set_ylim(-0.5, y_pos + 0.5)
plt.tight_layout()
fig4_path = FIGURES_DIR / "concordance_intervals.pdf"
plt.savefig(fig4_path, bbox_inches='tight')
plt.close()
print(f"✅ Figure 4 saved: {fig4_path}")

print("[5/6] Generating Figure 5: Monte Carlo Coverage...")

# Load MC results
with open(RESULTS_DIR / "monte_carlo_coverage.json") as f:
    mc_results = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, anchor_code in enumerate(['N', 'M', 'L']):
    ax = axes[idx]
    tensor = anchor_tensors[anchor_code]
    result = concordance[anchor_code]

    # Generate sample histogram (simulated for visualization)
    np.random.seed(20251012 + idx)
    samples = np.random.normal(tensor['H0_mean'], tensor['H0_uncertainty'], 10000)

    ax.hist(samples, bins=50, alpha=0.6, color='skyblue', edgecolor='black', density=True)

    # N/U bounds
    lower = result['n_merged'] - result['u_merged']
    upper = result['n_merged'] + result['u_merged']

    ax.axvline(lower, color='red', linestyle='--', linewidth=2, label=f'N/U bounds')
    ax.axvline(upper, color='red', linestyle='--', linewidth=2)

    # Coverage annotation
    coverage = mc_results['coverage']['by_anchor'][anchor_code]['coverage_fraction']
    ax.text(0.05, 0.95, f'Coverage: {100*coverage:.1f}%',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('H₀ (km/s/Mpc)')
    ax.set_ylabel('Density')
    ax.set_title(f"{tensor['anchor_name']}", fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.suptitle('Monte Carlo Validation: N/U Bounds Coverage', fontsize=14, fontweight='bold')
plt.tight_layout()
fig5_path = FIGURES_DIR / "monte_carlo_coverage.pdf"
plt.savefig(fig5_path, bbox_inches='tight')
plt.close()
print(f"✅ Figure 5 saved: {fig5_path}")

print("[6/6] Generating Figure 6: Epistemic Distance Components...")

# Load cross-anchor distances
with open(RESULTS_DIR / "cross_anchor_distances.json") as f:
    cross_distances = json.load(f)

fig, ax = plt.subplots(figsize=(10, 6))

pairs = list(cross_distances.keys())
n_pairs = len(pairs)
x = np.arange(n_pairs)
width = 0.6

# Prepare data
delta_0t = [cross_distances[p]['Delta_0_t'] for p in pairs]
delta_0m = [cross_distances[p]['Delta_0_m'] for p in pairs]
delta_0a = [cross_distances[p]['Delta_0_a'] for p in pairs]

# Stacked bar chart
ax.bar(x, delta_0t, width, label='Δ0_t (temporal)', color='skyblue')
ax.bar(x, delta_0m, width, bottom=delta_0t, label='Δ0_m (material)', color='lightcoral')
bottom = [delta_0t[i] + delta_0m[i] for i in range(n_pairs)]
ax.bar(x, delta_0a, width, bottom=bottom, label='Δ0_a (awareness)', color='lightgreen')

ax.set_ylabel('Epistemic Distance Component')
ax.set_xlabel('Anchor Pair')
ax.set_title('Epistemic Distance Components Across Anchors', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(pairs, rotation=15, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig6_path = FIGURES_DIR / "epistemic_components.pdf"
plt.savefig(fig6_path, bbox_inches='tight')
plt.close()
print(f"✅ Figure 6 saved: {fig6_path}")

# Final summary
print("\n" + "=" * 80)
print("✅ TASK 6 COMPLETE: PUBLICATION FIGURES GENERATED")
print("=" * 80)
print(f"\n   Generated 6 figures in {FIGURES_DIR}/:")
print(f"      1. covariance_heatmap.pdf")
print(f"      2. eigenspectrum.pdf")
print(f"      3. anchor_comparison.pdf")
print(f"      4. concordance_intervals.pdf")
print(f"      5. monte_carlo_coverage.pdf")
print(f"      6. epistemic_components.pdf")
print("=" * 80)
print("✅ Ready for summary report (TASK 7)")
print("=" * 80)
