#!/usr/bin/env python3
"""
CRITICAL DIAGNOSIS: Is the 97.4% reduction real or artifact?
"""

import numpy as np

print("="*70)
print("CRITICAL DIAGNOSIS: WHAT IS REALLY HAPPENING?")
print("="*70)

planck_n, planck_u = 67.40, 0.50
shoes_n, shoes_u = 73.64, 3.03

print("\n[1] INVERSE-VARIANCE WEIGHT CALCULATION")
print("-" * 70)
w_planck = 1.0 / planck_u**2
w_shoes = 1.0 / shoes_u**2
total_w = w_planck + w_shoes

print(f"Planck weight: {w_planck:.2f} (uncertainty: {planck_u})")
print(f"SH0ES weight:  {w_shoes:.2f} (uncertainty: {shoes_u})")
print(f"Weight ratio: Planck/SH0ES = {w_planck/w_shoes:.1f}:1")
print(f"\nPlanck contributes {100*w_planck/total_w:.1f}% of merged value")
print(f"SH0ES contributes {100*w_shoes/total_w:.1f}% of merged value")

print("\n[2] WHAT DETERMINES THE MERGED VALUE?")
print("-" * 70)
contribution_planck = (w_planck / total_w) * planck_n
contribution_shoes = (w_shoes / total_w) * shoes_n
merged = contribution_planck + contribution_shoes

print(f"Planck contribution: {contribution_planck:.2f} km/s/Mpc")
print(f"SH0ES contribution:  {contribution_shoes:.2f} km/s/Mpc")
print(f"Merged value:        {merged:.2f} km/s/Mpc")
print(f"\nMerged is {merged - planck_n:.2f} km/s/Mpc from Planck")
print(f"Merged is {shoes_n - merged:.2f} km/s/Mpc from SH0ES")

print("\n[3] THE PROBLEM: CHOICE OF BASELINE METRIC")
print("-" * 70)
print("Our claimed reduction uses 'disagreement between means':")
print(f"  Before: |{shoes_n} - {planck_n}| = {abs(shoes_n - planck_n):.2f}")
print(f"  After:  |{merged:.2f} - {planck_n}| = {abs(merged - planck_n):.2f}")
print(f"  Reduction: {100*(1 - abs(merged - planck_n)/abs(shoes_n - planck_n)):.1f}%")
print("\nBUT this is misleading because:")
print("  • SH0ES has huge uncertainty (3.03), so large mean offset is expected")
print("  • Merged value MUST be close to Planck (it has 16× more weight)")

print("\n[4] ALTERNATIVE METRIC: TENSION SIGNIFICANCE")
print("-" * 70)
# Tension is typically measured in sigma
tension_before = abs(shoes_n - planck_n) / np.sqrt(planck_u**2 + shoes_u**2)
tension_after = abs(merged - planck_n) / planck_u  # Compare merged to Planck

print(f"Before merge: {abs(shoes_n - planck_n):.2f} / √({planck_u}² + {shoes_u}²)")
print(f"            = {abs(shoes_n - planck_n):.2f} / {np.sqrt(planck_u**2 + shoes_u**2):.2f}")
print(f"            = {tension_before:.2f}σ tension")
print(f"\nAfter merge:  {abs(merged - planck_n):.2f} / {planck_u}")
print(f"            = {tension_after:.2f}σ tension")
print(f"\nTension reduction: {100*(1 - tension_after/tension_before):.1f}%")

print("\n[5] WHAT ABOUT THE ANCHOR SPREAD?")
print("-" * 70)
# The 51.9% anchor contribution is REAL
# But does it help resolve the tension with Planck?
print("Anchor means:")
print("  NGC4258:  72.52 km/s/Mpc")
print("  Milky Way: 76.17 km/s/Mpc")
print("  LMC:      72.27 km/s/Mpc")
print("  Spread:   3.90 km/s/Mpc")
print("\nThis spread is REAL and contributes 51.9% of total variance.")
print("BUT: All three anchors are still in tension with Planck (67.40)!")
print("  NGC4258 - Planck:  5.12 km/s/Mpc (10.2σ)")
print("  MW - Planck:       8.77 km/s/Mpc (17.5σ)")
print("  LMC - Planck:      4.87 km/s/Mpc (9.7σ)")

print("\n[6] THE OBSERVER TENSOR CONTRIBUTION")
print("-" * 70)
print("Observer tensor weighting produced:")
print("  NGC4258: 33.5% → mean 72.52")
print("  MW:      32.9% → mean 76.17")
print("  LMC:     33.5% → mean 72.27")
print(f"  Weighted: {0.335*72.52 + 0.329*76.17 + 0.335*72.27:.2f} ≈ 73.64")
print("\nThis is nearly equal weighting (33/33/33)!")
print("Observer tensors provided minimal differentiation.")

print("\n[7] WHAT IS THE EPISTEMIC PENALTY DOING?")
print("-" * 70)
base_unc = 0.493  # From inv-var merge
epistemic_penalty = np.sqrt(0.51**2 - 0.493**2)
print(f"Base uncertainty (inv-var): {base_unc:.3f} km/s/Mpc")
print(f"Epistemic penalty added:    {epistemic_penalty:.3f} km/s/Mpc")
print(f"Total uncertainty:          {0.51:.3f} km/s/Mpc")
print(f"\nEpistemic penalty is {100*epistemic_penalty/0.51:.1f}% of total uncertainty")
print("This is a small correction, not driving the result.")

print("\n" + "="*70)
print("VERDICT")
print("="*70)
print("\n❌ THE 97.4% REDUCTION IS MISLEADING")
print("\nWhat's actually happening:")
print("  1. Inverse-variance weighting gives Planck 94% of the weight")
print("  2. Merged value is forced to 67.57 (only 0.17 from Planck)")
print("  3. We then claim 'huge reduction' by comparing to pre-merge SH0ES")
print("  4. This is circular: of course the merge is close to Planck!")
print("\n✓ WHAT IS REAL:")
print("  • The 51.9% anchor contribution is real and stable")
print("  • The anchor spread (3.90 km/s/Mpc) is genuine")
print("  • Observer tensors correctly identify this structure")
print("\n❌ WHAT IS NOT REAL:")
print("  • The claim of 'resolving the Hubble tension'")
print("  • The 97.4% reduction (artifact of metric choice)")
print("  • Any evidence against new physics")
print("\n🎯 HONEST CLAIM:")
print("We identified that anchor-dependent systematics contribute 51.9% of")
print("SH0ES variance, but this does NOT resolve the tension with Planck.")
print("All three anchors remain in >9σ tension with CMB measurement.")
print("\n" + "="*70)
