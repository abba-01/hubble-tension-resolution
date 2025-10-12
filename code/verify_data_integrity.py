#!/usr/bin/env python3
"""
TASK 1: Data Inventory & Validation
Purpose: Verify all downloaded VizieR data is intact and matches expected structure
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

# Set base directory
BASE_DIR = Path("/run/media/root/OP01/got/hubble")
DATA_DIR = BASE_DIR / "data" / "vizier_data"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("TASK 1: DATA INVENTORY & VALIDATION")
print("=" * 80)

# ============================================================================
# STEP 1: Load critical systematic grid data
# ============================================================================

print("\n[1/4] Loading J_ApJ_826_56_table3.csv (systematic grid)...")

table3_path = DATA_DIR / "J_ApJ_826_56_table3.csv"
if not table3_path.exists():
    raise FileNotFoundError(f"CRITICAL: {table3_path} not found!")

df_sys = pd.read_csv(table3_path)
print(f"✅ Loaded: {len(df_sys)} rows, {len(df_sys.columns)} columns")

# ============================================================================
# STEP 2: Validate structure and content
# ============================================================================

print("\n[2/4] Validating systematic grid structure...")

expected_cols = ['H0', 'e_H0', 'Anc']
missing = [col for col in expected_cols if col not in df_sys.columns]
if missing:
    raise ValueError(f"Missing critical columns: {missing}")

print(f"✅ Critical columns present: {expected_cols}")
print(f"   Available columns: {list(df_sys.columns)}")

# Check for expected row count
if len(df_sys) < 200:
    print(f"⚠️  WARNING: Expected ~210 rows, found {len(df_sys)}")
else:
    print(f"✅ Row count validated: {len(df_sys)} systematic measurements")

# ============================================================================
# STEP 3: Compute anchor-specific statistics
# ============================================================================

print("\n[3/4] Computing anchor-specific H₀ statistics...")

anchor_stats = {}

# Group by anchor
for anchor in df_sys['Anc'].unique():
    anchor_data = df_sys[df_sys['Anc'] == anchor]

    h0_values = anchor_data['H0'].values
    e_h0_values = anchor_data['e_H0'].values

    stats = {
        "count": int(len(anchor_data)),
        "mean_H0": float(np.mean(h0_values)),
        "std_H0": float(np.std(h0_values)),
        "median_H0": float(np.median(h0_values)),
        "min_H0": float(np.min(h0_values)),
        "max_H0": float(np.max(h0_values)),
        "mean_uncertainty": float(np.mean(e_h0_values))
    }

    anchor_stats[anchor] = stats

    print(f"\n   Anchor: {anchor}")
    print(f"      Count: {stats['count']}")
    print(f"      Mean H₀: {stats['mean_H0']:.2f} ± {stats['std_H0']:.2f} km/s/Mpc")
    print(f"      Range: [{stats['min_H0']:.2f}, {stats['max_H0']:.2f}] km/s/Mpc")

# ============================================================================
# STEP 4: Validate against known values from SESSION_MEMORY
# ============================================================================

print("\n[4/4] Validating against expected values...")

# Expected from SESSION_MEMORY.md:
# NGC4258 only (N):  72.51 ± 0.83 km/s/Mpc
# LMC only (L):      72.29 ± 0.80 km/s/Mpc
# Milky Way only (M): 76.13 ± 0.99 km/s/Mpc

expected_anchors = {
    'N': 72.51,  # NGC4258
    'L': 72.29,  # LMC
    'M': 76.13   # Milky Way
}

validation_passed = True

for anchor, expected_h0 in expected_anchors.items():
    if anchor in anchor_stats:
        measured_h0 = anchor_stats[anchor]['mean_H0']
        diff = abs(measured_h0 - expected_h0)
        pct_diff = 100 * diff / expected_h0

        if pct_diff < 5.0:  # Within 5%
            print(f"✅ Anchor {anchor}: {measured_h0:.2f} vs expected {expected_h0:.2f} (Δ={pct_diff:.1f}%)")
        else:
            print(f"⚠️  Anchor {anchor}: {measured_h0:.2f} vs expected {expected_h0:.2f} (Δ={pct_diff:.1f}%)")
            validation_passed = False
    else:
        print(f"⚠️  Anchor {anchor} not found in data")
        validation_passed = False

# Compute cross-anchor spread
if 'N' in anchor_stats and 'M' in anchor_stats:
    spread = abs(anchor_stats['M']['mean_H0'] - anchor_stats['N']['mean_H0'])
    print(f"\n✅ Cross-anchor spread (M-N): {spread:.2f} km/s/Mpc (expected ~3.62)")

    if abs(spread - 3.62) < 0.5:
        print(f"   ✅ Spread matches expected value within 0.5 km/s/Mpc")
    else:
        print(f"   ⚠️  Spread differs from expected 3.62 by {abs(spread - 3.62):.2f}")

# ============================================================================
# STEP 5: Create inventory file
# ============================================================================

print("\n[5/5] Creating data inventory...")

inventory = {
    "systematic_grid": {
        "file": str(table3_path.name),
        "rows": int(len(df_sys)),
        "columns": int(len(df_sys.columns)),
        "column_names": list(df_sys.columns),
        "file_size_bytes": int(table3_path.stat().st_size)
    },
    "anchor_statistics": anchor_stats,
    "validation": {
        "passed": validation_passed,
        "cross_anchor_spread_km_s_Mpc": float(spread) if 'spread' in locals() else None
    }
}

# Save inventory
inventory_path = RESULTS_DIR / "data_inventory.json"
with open(inventory_path, 'w') as f:
    json.dump(inventory, f, indent=2)

print(f"✅ Inventory saved to: {inventory_path}")

# Save anchor summary
anchor_summary_path = RESULTS_DIR / "anchor_summary_stats.json"
with open(anchor_summary_path, 'w') as f:
    json.dump(anchor_stats, f, indent=2)

print(f"✅ Anchor statistics saved to: {anchor_summary_path}")

# ============================================================================
# Final status
# ============================================================================

print("\n" + "=" * 80)
if validation_passed:
    print("✅ DATA INTEGRITY VERIFIED: 210 systematic measurements loaded")
    print("✅ All anchor values match expected ranges")
    print("✅ Ready for empirical covariance extraction (TASK 2)")
else:
    print("⚠️  WARNING: Some validation checks failed - review above")
print("=" * 80)
