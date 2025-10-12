#!/usr/bin/env python3
"""
TRACE THE ACTUAL CALCULATION THAT PRODUCED 67.57 ± 0.51
Show exactly what the code did vs what it should have done
"""

import numpy as np
import json
from pathlib import Path

BASE_DIR = Path("/run/media/root/OP01/got/hubble")

print("="*80)
print("TRACING THE ACTUAL CALCULATION")
print("="*80)

# Load actual MCMC samples
sN = np.load(BASE_DIR / 'data/mast/mcmc_samples_N.npy')
sM = np.load(BASE_DIR / 'data/mast/mcmc_samples_M.npy')
sL = np.load(BASE_DIR / 'data/mast/mcmc_samples_L.npy')

# Load actual observer tensors
with open(BASE_DIR / "results/anchor_tensors.json") as f:
    tensors = json.load(f)

print("\n[1] ACTUAL SYSTEMATIC_FRACTION VALUES IN TENSORS")
print("-"*80)
for code, data in tensors.items():
    sys_frac = data['systematic_fraction']
    print(f"{data['anchor_name']}: systematic_fraction = {sys_frac}")

print("\n⚠️  ALL VALUES ARE 0.0!")
print("This means the systematic_fraction correction was NOT applied.")

print("\n[2] ACTUAL OBSERVER WEIGHTS")
print("-"*80)

# Observer tensor weights (from actual code)
Delta_T_values = {code: data['tensor_magnitude'] for code, data in tensors.items()}
weights = {}
total_w = 0.0

for code, Delta_T in Delta_T_values.items():
    w = 1.0 / (1.0 + Delta_T)
    weights[code] = w
    total_w += w

# Normalize
for code in weights:
    weights[code] /= total_w

print("Observer tensor weights:")
for code, data in tensors.items():
    print(f"  {data['anchor_name']}: {100*weights[code]:.1f}%  (Δ_T = {Delta_T_values[code]:.4f})")

print("\n[3] ACTUAL WEIGHTED LATE-TIME POSTERIOR")
print("-"*80)

# Compute weighted statistics (actual code path)
all_samples = []
all_weights = []

for code, samples in [('N', sN), ('M', sM), ('L', sL)]:
    w = weights[code]
    all_samples.extend(samples)
    all_weights.extend([w / len(samples)] * len(samples))

all_samples = np.array(all_samples)
all_weights = np.array(all_weights)
all_weights /= np.sum(all_weights)

late_h0 = np.sum(all_samples * all_weights)
late_var = np.sum(all_weights * (all_samples - late_h0)**2)
late_std = np.sqrt(late_var)

print(f"Weighted late-time: {late_h0:.4f} ± {late_std:.4f} km/s/Mpc")

print("\n[4] ACTUAL MERGE CALCULATION")
print("-"*80)

planck_h0 = 67.40
planck_unc = 0.50

# Inverse-variance weights
w_early = 1.0 / planck_unc**2
w_late = 1.0 / late_std**2

merged_h0 = (w_early * planck_h0 + w_late * late_h0) / (w_early + w_late)
merged_base = 1.0 / np.sqrt(w_early + w_late)

print(f"Base merge: {merged_h0:.4f} ± {merged_base:.4f} km/s/Mpc")

# ACTUAL epistemic penalty (what the code did)
Delta_T_avg = np.mean(list(Delta_T_values.values()))
disagreement = abs(planck_h0 - late_h0)
epistemic_factor = 0.1

penalty_actual = (disagreement / 3) * Delta_T_avg * epistemic_factor

print(f"\nActual epistemic penalty calculation:")
print(f"  disagreement = {disagreement:.4f}")
print(f"  Delta_T_avg  = {Delta_T_avg:.4f}")
print(f"  factor       = {epistemic_factor}")
print(f"  penalty = ({disagreement:.4f} / 3) × {Delta_T_avg:.4f} × {epistemic_factor}")
print(f"          = {penalty_actual:.4f} km/s/Mpc")

merged_actual = np.sqrt(merged_base**2 + penalty_actual**2)

print(f"\nActual result: {merged_h0:.4f} ± {merged_actual:.4f} km/s/Mpc")
print(f"Reported:      67.5652 ± 0.5053 km/s/Mpc")
print(f"Match: {abs(merged_h0 - 67.5652) < 0.001 and abs(merged_actual - 0.5053) < 0.001}")

print("\n" + "="*80)
print("WHAT THE CODE SHOULD HAVE DONE")
print("="*80)

# Compute actual systematic_fraction from variance decomposition
anchor_means = [sN.mean(), sM.mean(), sL.mean()]
anchor_variance = np.var(anchor_means, ddof=1)
total_variance = np.var(all_samples, ddof=1)
systematic_fraction = anchor_variance / total_variance

print(f"\nCompute systematic_fraction from data:")
print(f"  anchor_variance = Var({sN.mean():.2f}, {sM.mean():.2f}, {sL.mean():.2f})")
print(f"                  = {anchor_variance:.4f}")
print(f"  total_variance  = {total_variance:.4f}")
print(f"  systematic_fraction = {systematic_fraction:.4f} ({100*systematic_fraction:.1f}%)")

# CORRECT epistemic penalty
penalty_correct = (disagreement / 2) * Delta_T_avg * (1 - systematic_fraction)

print(f"\nCorrect epistemic penalty:")
print(f"  penalty = (disagreement / 2) × Δ_T × (1 - systematic_fraction)")
print(f"          = ({disagreement:.4f} / 2) × {Delta_T_avg:.4f} × (1 - {systematic_fraction:.4f})")
print(f"          = {disagreement/2:.4f} × {Delta_T_avg:.4f} × {1-systematic_fraction:.4f}")
print(f"          = {penalty_correct:.4f} km/s/Mpc")

merged_correct = np.sqrt(merged_base**2 + penalty_correct**2)

print(f"\nCorrect result: {merged_h0:.4f} ± {merged_correct:.4f} km/s/Mpc")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)

print(f"\nWhat we reported:  67.57 ± 0.51 km/s/Mpc")
print(f"What we computed:  {merged_h0:.2f} ± {merged_actual:.2f} km/s/Mpc ✓")
print(f"What we should've: {merged_h0:.2f} ± {merged_correct:.2f} km/s/Mpc")

print(f"\nUncertainty underestimated by: {merged_correct - merged_actual:.2f} km/s/Mpc")
print(f"Factor: {merged_correct / merged_actual:.2f}×")

print("\n" + "="*80)
print("ROOT CAUSE")
print("="*80)

print("\n1. systematic_fraction in tensors.json = 0.0 (wrong)")
print(f"   Should be: {systematic_fraction:.3f} (from variance decomposition)")

print("\n2. Merge code doesn't use systematic_fraction at all")
print("   Uses arbitrary 'factor = 0.1' instead")

print("\n3. Formula has arbitrary divisions:")
print("   • disagreement / 3 (should be / 2)")
print("   • factor = 0.1 (should be (1 - systematic_fraction) = 0.481)")

print("\n4. Result: Epistemic penalty too small by factor of 7.2×")
print(f"   Actual:  {penalty_actual:.4f} km/s/Mpc")
print(f"   Correct: {penalty_correct:.4f} km/s/Mpc")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

print("\nThe systematic_fraction correction is:")
print("  • Documented in your framework ✓")
print("  • Present in code structure (tensors have the field) ✓")
print("  • But NOT computed from data (set to 0.0) ✗")
print("  • And NOT used in merge formula ✗")

print("\nInstead, arbitrary constants (/ 3, × 0.1) were used.")
print("These happened to give a similar-looking result (0.51 vs 0.93),")
print("but the underlying calculation is wrong.")

print("\n🎯 TO FIX: Use the 51.9% systematic_fraction in the merge formula!")

print("\n" + "="*80)
