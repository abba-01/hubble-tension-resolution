#!/usr/bin/env python3
"""
TEST: What happens with the CORRECT epistemic penalty formula?
"""

import numpy as np

print("="*80)
print("TESTING CORRECT EPISTEMIC PENALTY FORMULA")
print("="*80)

# Inputs
planck_n, planck_u = 67.40, 0.50
shoes_n, shoes_u = 73.64, 3.03
disagreement = abs(planck_n - shoes_n)

# Inverse-variance merge base
w_planck = 1.0 / planck_u**2
w_shoes = 1.0 / shoes_u**2
merged_base = (w_planck * planck_n + w_shoes * shoes_n) / (w_planck + w_shoes)
sigma_base = 1.0 / np.sqrt(w_planck + w_shoes)

print("\n[BASE] Inverse-Variance Merge:")
print(f"  Merged: {merged_base:.4f} ± {sigma_base:.4f} km/s/Mpc")

# Observer tensor average
Delta_T_avg = 0.5266

# CRITICAL: systematic_fraction from variance decomposition
systematic_fraction = 0.519  # 51.9% anchor contribution

print("\n" + "="*80)
print("WRONG FORMULA (what we used):")
print("="*80)

epistemic_factor = 0.1
penalty_wrong = (disagreement / 3) * Delta_T_avg * epistemic_factor

print(f"  epistemic_penalty = (disagreement / 3) × Δ_T × factor")
print(f"                    = ({disagreement:.2f} / 3) × {Delta_T_avg:.4f} × {epistemic_factor}")
print(f"                    = {penalty_wrong:.4f} km/s/Mpc")

sigma_wrong = np.sqrt(sigma_base**2 + penalty_wrong**2)
print(f"\n  Total uncertainty: √({sigma_base:.4f}² + {penalty_wrong:.4f}²)")
print(f"                   = {sigma_wrong:.4f} km/s/Mpc")
print(f"\n  RESULT: {merged_base:.2f} ± {sigma_wrong:.2f} km/s/Mpc")

print("\n" + "="*80)
print("CORRECT FORMULA (from your framework document):")
print("="*80)

penalty_correct = (disagreement / 2) * Delta_T_avg * (1 - systematic_fraction)

print(f"  epistemic_penalty = (disagreement / 2) × Δ_T × (1 - systematic_fraction)")
print(f"                    = ({disagreement:.2f} / 2) × {Delta_T_avg:.4f} × (1 - {systematic_fraction:.3f})")
print(f"                    = {disagreement/2:.4f} × {Delta_T_avg:.4f} × {1 - systematic_fraction:.3f}")
print(f"                    = {penalty_correct:.4f} km/s/Mpc")

sigma_correct = np.sqrt(sigma_base**2 + penalty_correct**2)
print(f"\n  Total uncertainty: √({sigma_base:.4f}² + {penalty_correct:.4f}²)")
print(f"                   = {sigma_correct:.4f} km/s/Mpc")
print(f"\n  RESULT: {merged_base:.2f} ± {sigma_correct:.2f} km/s/Mpc")

print("\n" + "="*80)
print("COMPARISON:")
print("="*80)

print(f"\nWrong formula penalty:  {penalty_wrong:.4f} km/s/Mpc")
print(f"Correct formula penalty: {penalty_correct:.4f} km/s/Mpc")
print(f"Ratio: {penalty_correct/penalty_wrong:.2f}× larger")

print(f"\nWrong total uncertainty:  {sigma_wrong:.4f} km/s/Mpc")
print(f"Correct total uncertainty: {sigma_correct:.4f} km/s/Mpc")
print(f"Difference: {sigma_correct - sigma_wrong:.4f} km/s/Mpc")

print("\n" + "="*80)
print("INTERPRETATION:")
print("="*80)

print("\nThe CORRECT formula says:")
print(f"  • 51.9% of disagreement is already accounted for in anchor variance")
print(f"  • Only penalize for the REMAINING 48.1% of epistemic distance")
print(f"  • This gives penalty = {penalty_correct:.4f} km/s/Mpc")
print(f"  • Total uncertainty = {sigma_correct:.4f} km/s/Mpc")

print("\nThe WRONG formula says:")
print(f"  • Ignore systematic_fraction entirely")
print(f"  • Arbitrarily divide by 3 instead of 2")
print(f"  • Add arbitrary 'factor = 0.1' to make it smaller")
print(f"  • This gives penalty = {penalty_wrong:.4f} km/s/Mpc (too small!)")

print("\n" + "="*80)
print("WHAT ABOUT THE ALTERNATIVE CORRECTION?")
print("="*80)

# Maybe Δ_T itself should be reduced?
# If systematic_fraction is high, epistemic distance is LESS than we thought

Delta_T_effective = Delta_T_avg * (1 - systematic_fraction)
penalty_alt = (disagreement / 2) * Delta_T_effective

print(f"\nAlternative: Reduce Δ_T by systematic_fraction")
print(f"  Δ_T_effective = {Delta_T_avg:.4f} × (1 - {systematic_fraction:.3f})")
print(f"                = {Delta_T_effective:.4f}")
print(f"\n  epistemic_penalty = (disagreement / 2) × Δ_T_effective")
print(f"                    = ({disagreement:.2f} / 2) × {Delta_T_effective:.4f}")
print(f"                    = {penalty_alt:.4f} km/s/Mpc")

sigma_alt = np.sqrt(sigma_base**2 + penalty_alt**2)
print(f"\n  Total uncertainty: {sigma_alt:.4f} km/s/Mpc")
print(f"  RESULT: {merged_base:.2f} ± {sigma_alt:.2f} km/s/Mpc")

print("\n" + "="*80)
print("FINAL VERDICT:")
print("="*80)

print("\n❌ WHAT WE USED:")
print(f"   {merged_base:.2f} ± {sigma_wrong:.2f} km/s/Mpc")
print("   • Missing systematic_fraction term")
print("   • Arbitrary divisions (/ 3, × 0.1)")

print("\n✅ CORRECT FORMULA (Option 1):")
print(f"   {merged_base:.2f} ± {sigma_correct:.2f} km/s/Mpc")
print("   • penalty × (1 - systematic_fraction)")

print("\n✅ CORRECT FORMULA (Option 2):")
print(f"   {merged_base:.2f} ± {sigma_alt:.2f} km/s/Mpc")
print("   • Δ_T × (1 - systematic_fraction)")

print("\nBoth correct formulas give LARGER uncertainty than what we reported!")
print(f"We claimed: 67.57 ± 0.51 km/s/Mpc")
print(f"Should be:  67.57 ± {sigma_correct:.2f} km/s/Mpc (or ± {sigma_alt:.2f})")

print("\n🎯 THE BUG:")
print("We forgot to apply the systematic_fraction reduction!")
print("This made our merged uncertainty too small (0.51 vs ~0.8-1.0).")

print("\n" + "="*80)
