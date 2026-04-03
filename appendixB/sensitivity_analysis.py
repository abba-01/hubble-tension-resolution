#!/usr/bin/env python3
"""
Sensitivity Analysis: ΔT Impact on Hubble Tension Resolution
Critical question: What is the minimum ΔT for 100% concordance?
"""

import numpy as np
import pandas as pd

# Fixed parameters from SSoT
n_merged = 69.9981  # km/s/Mpc
u_std = 0.6496      # km/s/Mpc
disagreement = 5.3527  # km/s/Mpc
late_upper = 73.5773   # km/s/Mpc (critical boundary)

# Sensitivity grid
delta_T_range = np.arange(1.0, 1.65, 0.05)

results = []

print("="*70)
print("ΔT SENSITIVITY ANALYSIS")
print("="*70)
print(f"Fixed parameters:")
print(f"  n_merged     = {n_merged:.4f} km/s/Mpc")
print(f"  u_std        = {u_std:.4f} km/s/Mpc")
print(f"  disagreement = {disagreement:.4f} km/s/Mpc")
print(f"  late_upper   = {late_upper:.4f} km/s/Mpc (critical boundary)")
print("="*70)
print()

critical_deltaT = None

for delta_T in delta_T_range:
    # Compute tensor expansion
    u_expand = (disagreement / 2) * delta_T
    
    # Total merged uncertainty
    u_merged = u_std + u_expand
    
    # Merged interval boundaries
    merged_lower = n_merged - u_merged
    merged_upper = n_merged + u_merged
    
    # Gap calculation (only late_upper matters for concordance)
    gap = max(0, late_upper - merged_upper)
    
    # Resolution percentage
    resolution_pct = (1 - gap/disagreement) * 100
    
    # Containment status
    full_concordance = (gap == 0)
    
    results.append({
        'ΔT': delta_T,
        'u_expand': u_expand,
        'u_merged': u_merged,
        'merged_lower': merged_lower,
        'merged_upper': merged_upper,
        'gap': gap,
        'resolution_%': resolution_pct,
        'concordance': '✓' if full_concordance else '✗'
    })
    
    # Find critical ΔT (first where gap = 0)
    if critical_deltaT is None and full_concordance:
        critical_deltaT = delta_T

# Create DataFrame
df = pd.DataFrame(results)

# Display results
print("SENSITIVITY GRID RESULTS:")
print("-"*70)
print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
print("-"*70)
print()

# Critical analysis
if critical_deltaT is not None:
    print("CRITICAL THRESHOLD ANALYSIS:")
    print("="*70)
    print(f"Critical ΔT = {critical_deltaT:.2f}")
    print(f"  (Minimum epistemic distance for 100% concordance)")
    print()
    
    empirical_deltaT = 1.3477
    stability_margin = (empirical_deltaT - critical_deltaT) / critical_deltaT * 100
    
    print(f"Empirical ΔT = {empirical_deltaT:.4f}")
    print(f"Stability margin = {stability_margin:.1f}%")
    print()
    
    if stability_margin > 20:
        print("✓ ROBUST: >20% margin provides strong stability")
    elif stability_margin > 10:
        print("⚠ MODERATE: 10-20% margin, reasonable but not excessive")
    else:
        print("✗ FRAGILE: <10% margin, sensitive to tensor calibration")
    
    print("="*70)
    print()
    
    # Analytic verification
    print("ANALYTIC VERIFICATION:")
    print("-"*70)
    gap_to_close = late_upper - (n_merged + u_std)
    deltaT_analytic = 2 * gap_to_close / disagreement
    print(f"Gap to close: {gap_to_close:.4f} km/s/Mpc")
    print(f"Required u_expand: {gap_to_close:.4f} km/s/Mpc")
    print(f"Analytic ΔT = 2 × {gap_to_close:.4f} / {disagreement:.4f}")
    print(f"            = {deltaT_analytic:.4f}")
    print(f"Grid ΔT     = {critical_deltaT:.2f}")
    print(f"Match: {'✓' if abs(deltaT_analytic - critical_deltaT) < 0.05 else '✗'}")
    print("="*70)
else:
    print("⚠ WARNING: No ΔT in range [1.0, 1.6] achieves concordance!")
    print("Maximum resolution: {:.1f}%".format(df['resolution_%'].max()))

# Summary statistics
print()
print("SUMMARY STATISTICS:")
print("="*70)
print(f"ΔT range tested: [{delta_T_range[0]:.2f}, {delta_T_range[-1]:.2f}]")
print(f"Resolution range: [{df['resolution_%'].min():.1f}%, {df['resolution_%'].max():.1f}%]")
print(f"Maximum gap: {df['gap'].max():.4f} km/s/Mpc")
print(f"Minimum gap: {df['gap'].min():.4f} km/s/Mpc")
print("="*70)

# Export results
output_file = "deltaT_sensitivity_results.csv"
df.to_csv(output_file, index=False)
print(f"\n✓ Results saved to: {output_file}")
