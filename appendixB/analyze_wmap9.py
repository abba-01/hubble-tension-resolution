#!/usr/bin/env python3
"""
Analyze WMAP9 observer tensor and compare with Planck/SH0ES
"""

import numpy as np
import json

# WMAP9 + BAO + H₀ parameters
wmap9 = {
    "H0": 69.61,
    "H0_err": 0.91,
    "Omega_m": 0.285,
    "Omega_Lambda": 0.715,
    "source": "WMAP9+BAO+H0"
}

# Compute WMAP9 observer tensor
a_wmap9 = 1.0 - (wmap9["H0_err"] / wmap9["H0"])
T_wmap9 = np.array([a_wmap9, 0.85, 0.0, -0.4])

print("="*70)
print("WMAP9 OBSERVER TENSOR ANALYSIS")
print("="*70)
print()

print(f"H₀: {wmap9['H0']} ± {wmap9['H0_err']} km/s/Mpc")
print(f"Relative precision: {(wmap9['H0_err']/wmap9['H0'])*100:.2f}%")
print()

print(f"Observer tensor: {T_wmap9}")
print(f"  a   = {T_wmap9[0]:.4f} (awareness)")
print(f"  P_m = {T_wmap9[1]:.4f} (physics model)")
print(f"  0_t = {T_wmap9[2]:.4f} (temporal)")
print(f"  0_a = {T_wmap9[3]:.4f} (analysis)")
print()

# Compare with Planck and SH0ES
T_planck = np.array([0.9926, 0.95, 0.0, -0.5])
T_shoes = np.array([0.9305, 0.05, -0.05, 0.5])

print("EPISTEMIC DISTANCES:")
print("-"*70)

# WMAP9 vs SH0ES
delta_wmap_shoes = np.linalg.norm(T_wmap9 - T_shoes)
print(f"WMAP9 ↔ SH0ES:  ΔT = {delta_wmap_shoes:.4f}")

# Planck vs SH0ES (original)
delta_planck_shoes = np.linalg.norm(T_planck - T_shoes)
print(f"Planck ↔ SH0ES: ΔT = {delta_planck_shoes:.4f} (original)")

# WMAP9 vs Planck
delta_wmap_planck = np.linalg.norm(T_wmap9 - T_planck)
print(f"WMAP9 ↔ Planck: ΔT = {delta_wmap_planck:.4f}")
print()

print("INTERPRETATION:")
print("-"*70)
print(f"WMAP9 reduces epistemic distance by: {(1 - delta_wmap_shoes/delta_planck_shoes)*100:.1f}%")
print(f"WMAP9 H₀ bridges the gap: Planck (67.4) < WMAP9 (69.61) < SH0ES (73.04)")
print()

# Recompute merged uncertainty with WMAP9
disagreement = 5.3527  # km/s/Mpc (early vs late)
u_std = 0.6496  # km/s/Mpc
u_expand_wmap9 = (disagreement / 2) * delta_wmap_shoes
u_merged_wmap9 = u_std + u_expand_wmap9

print("MERGED UNCERTAINTY (WMAP9-based):")
print("-"*70)
print(f"Standard:     {u_std:.4f} km/s/Mpc")
print(f"Expansion:    {u_expand_wmap9:.4f} km/s/Mpc")
print(f"Total merged: {u_merged_wmap9:.4f} km/s/Mpc")
print()
print(f"Original (Planck-based): {4.26:.2f} km/s/Mpc")
print(f"WMAP9-based:             {u_merged_wmap9:.2f} km/s/Mpc")
print(f"Change:                  {u_merged_wmap9 - 4.26:.2f} km/s/Mpc ({((u_merged_wmap9/4.26 - 1)*100):.1f}%)")
print("="*70)

# Save results
results = {
    "wmap9_tensor": T_wmap9.tolist(),
    "epistemic_distances": {
        "wmap9_shoes": float(delta_wmap_shoes),
        "planck_shoes": float(delta_planck_shoes),
        "wmap9_planck": float(delta_wmap_planck)
    },
    "merged_uncertainty": {
        "original_planck": 4.26,
        "wmap9_based": float(u_merged_wmap9),
        "difference_km_s_Mpc": float(u_merged_wmap9 - 4.26)
    },
    "wmap9_parameters": wmap9
}

with open("wmap9_analysis.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✓ Results saved to wmap9_analysis.json")
