#!/usr/bin/env python3
"""
Monte Carlo Validation: Hubble Tension Concordance Frequency
Tests whether deterministic framework matches stochastic behavior
"""

import numpy as np
import matplotlib.pyplot as plt

def aggregate(H0_values, sigma_values):
    """
    N/U algebra weighted aggregation with inverse-variance weighting
    
    Returns:
        n: weighted nominal
        u: combined uncertainty
    """
    H0 = np.array(H0_values)
    sigma = np.array(sigma_values)
    
    # Inverse-variance weights
    w = 1.0 / np.square(sigma)
    weights = w / np.sum(w)
    
    # Weighted nominal
    n = np.sum(weights * H0)
    
    # Combined uncertainty
    u = np.sqrt(np.sum(np.square(weights) * np.square(sigma)))
    
    return n, u

# Fixed parameters from SSoT
merged_low = 65.7421
merged_high = 74.2541

# Published H₀ measurements (mean, sigma)
planck_params = (67.4, 0.5)
des_params = (67.19, 0.65)
shoes_params = (73.04, 1.04)
trgb_params = (69.8, 2.5)
tdcosmo_params = (73.3, 5.8)
maser_params = (73.5, 3.0)

# Monte Carlo parameters
N_samples = 10000
random_seed = 20251012  # For reproducibility

print("="*70)
print("MONTE CARLO CONCORDANCE VALIDATION")
print("="*70)
print(f"Samples: {N_samples:,}")
print(f"Random seed: {random_seed}")
print(f"Merged interval: [{merged_low:.4f}, {merged_high:.4f}] km/s/Mpc")
print("="*70)
print()

# Set random seed for reproducibility
np.random.seed(random_seed)

# Storage for results
contain_count = 0
early_intervals = []
late_intervals = []
early_contained = []
late_contained = []
both_contained = []

print("Running Monte Carlo simulation...")

for i in range(N_samples):
    # Draw random samples from each probe (Gaussian assumption)
    planck = np.random.normal(*planck_params)
    des = np.random.normal(*des_params)
    shoes = np.random.normal(*shoes_params)
    trgb = np.random.normal(*trgb_params)
    tdcosmo = np.random.normal(*tdcosmo_params)
    maser = np.random.normal(*maser_params)
    
    # Aggregate early universe probes
    early_n, early_u = aggregate([planck, des], 
                                  [planck_params[1], des_params[1]])
    
    # Aggregate late universe probes
    late_n, late_u = aggregate([shoes, trgb, tdcosmo, maser],
                                [shoes_params[1], trgb_params[1], 
                                 tdcosmo_params[1], maser_params[1]])
    
    # Compute intervals
    early_interval = [early_n - early_u, early_n + early_u]
    late_interval = [late_n - late_u, late_n + late_u]
    
    # Check containment
    early_in = (early_interval[0] >= merged_low and 
                early_interval[1] <= merged_high)
    late_in = (late_interval[0] >= merged_low and 
               late_interval[1] <= merged_high)
    both_in = early_in and late_in
    
    # Record results
    early_intervals.append(early_interval)
    late_intervals.append(late_interval)
    early_contained.append(early_in)
    late_contained.append(late_in)
    both_contained.append(both_in)
    
    if both_in:
        contain_count += 1
    
    # Progress indicator
    if (i + 1) % 1000 == 0:
        print(f"  {i+1:,} / {N_samples:,} samples processed...")

print()
print("="*70)
print("RESULTS")
print("="*70)

# Calculate frequencies
freq_early = sum(early_contained) / N_samples
freq_late = sum(late_contained) / N_samples
freq_both = contain_count / N_samples

print(f"Early universe containment: {freq_early:.4f} ({freq_early*100:.2f}%)")
print(f"Late universe containment:  {freq_late:.4f} ({freq_late*100:.2f}%)")
print(f"Both contained (concordance): {freq_both:.4f} ({freq_both*100:.2f}%)")
print()

# Interpretation
if freq_both >= 0.95:
    status = "✓ EXCELLENT"
    interpretation = "Deterministic framework aligns with probabilistic behavior"
elif freq_both >= 0.90:
    status = "✓ GOOD"
    interpretation = "Strong alignment, minor probabilistic leakage acceptable"
elif freq_both >= 0.80:
    status = "⚠ MODERATE"
    interpretation = "Reasonable but conservative bounds may need refinement"
else:
    status = "✗ POOR"
    interpretation = "Significant mismatch between deterministic and probabilistic"

print(f"Status: {status}")
print(f"Interpretation: {interpretation}")
print("="*70)
print()

# Statistical analysis
early_intervals_array = np.array(early_intervals)
late_intervals_array = np.array(late_intervals)

early_lower_mean = np.mean(early_intervals_array[:, 0])
early_upper_mean = np.mean(early_intervals_array[:, 1])
late_lower_mean = np.mean(late_intervals_array[:, 0])
late_upper_mean = np.mean(late_intervals_array[:, 1])

print("STATISTICAL SUMMARY:")
print("="*70)
print("Early universe intervals:")
print(f"  Mean lower bound: {early_lower_mean:.4f} km/s/Mpc")
print(f"  Mean upper bound: {early_upper_mean:.4f} km/s/Mpc")
print(f"  Mean width: {early_upper_mean - early_lower_mean:.4f} km/s/Mpc")
print()
print("Late universe intervals:")
print(f"  Mean lower bound: {late_lower_mean:.4f} km/s/Mpc")
print(f"  Mean upper bound: {late_upper_mean:.4f} km/s/Mpc")
print(f"  Mean width: {late_upper_mean - late_lower_mean:.4f} km/s/Mpc")
print()
print("Comparison with merged interval:")
print(f"  Merged: [{merged_low:.4f}, {merged_high:.4f}]")
print(f"  Width: {merged_high - merged_low:.4f} km/s/Mpc")
print("="*70)
print()

# Coverage analysis
early_lower_coverage = np.sum(early_intervals_array[:, 0] >= merged_low) / N_samples
early_upper_coverage = np.sum(early_intervals_array[:, 1] <= merged_high) / N_samples
late_lower_coverage = np.sum(late_intervals_array[:, 0] >= merged_low) / N_samples
late_upper_coverage = np.sum(late_intervals_array[:, 1] <= merged_high) / N_samples

print("BOUNDARY COVERAGE ANALYSIS:")
print("="*70)
print("Early universe:")
print(f"  Lower bounds inside merged: {early_lower_coverage:.4f}")
print(f"  Upper bounds inside merged: {early_upper_coverage:.4f}")
print()
print("Late universe:")
print(f"  Lower bounds inside merged: {late_lower_coverage:.4f}")
print(f"  Upper bounds inside merged: {late_upper_coverage:.4f}")
print("="*70)
print()

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Interval distribution
ax1 = axes[0]
ax1.hist([early_intervals_array[:, 0], early_intervals_array[:, 1]], 
         bins=50, alpha=0.6, label=['Early lower', 'Early upper'])
ax1.hist([late_intervals_array[:, 0], late_intervals_array[:, 1]], 
         bins=50, alpha=0.6, label=['Late lower', 'Late upper'])
ax1.axvline(merged_low, color='red', linestyle='--', linewidth=2, label='Merged bounds')
ax1.axvline(merged_high, color='red', linestyle='--', linewidth=2)
ax1.set_xlabel('H₀ (km/s/Mpc)')
ax1.set_ylabel('Frequency')
ax1.set_title('MC Interval Distribution vs Merged Bounds')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Containment frequency
ax2 = axes[1]
categories = ['Early only', 'Late only', 'Both\n(Concordance)']
frequencies = [freq_early, freq_late, freq_both]
colors = ['lightblue', 'lightcoral', 'lightgreen']
bars = ax2.bar(categories, frequencies, color=colors, edgecolor='black')
ax2.axhline(0.95, color='red', linestyle='--', linewidth=2, label='95% threshold')
ax2.set_ylabel('Containment Frequency')
ax2.set_ylim([0, 1.0])
ax2.set_title('Containment Frequencies')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, freq in zip(bars, frequencies):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{freq:.3f}\n({freq*100:.1f}%)',
             ha='center', va='bottom')

plt.tight_layout()
plt.savefig('monte_carlo_validation.png', dpi=150, bbox_inches='tight')
print("✓ Visualization saved to: monte_carlo_validation.png")
print()

# Export detailed results
results_summary = {
    'n_samples': N_samples,
    'random_seed': random_seed,
    'merged_interval': [merged_low, merged_high],
    'early_containment': freq_early,
    'late_containment': freq_late,
    'both_containment': freq_both,
    'early_mean_interval': [early_lower_mean, early_upper_mean],
    'late_mean_interval': [late_lower_mean, late_upper_mean],
    'status': status,
    'interpretation': interpretation
}

import json
with open('monte_carlo_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("✓ Results saved to: monte_carlo_results.json")
print()
print("="*70)
print("MONTE CARLO VALIDATION COMPLETE")
print("="*70)
