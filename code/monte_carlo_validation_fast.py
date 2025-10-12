#!/usr/bin/env python3
"""
TASK 5: Monte Carlo Validation (Fast Version)
Purpose: Verify N/U bounds are conservative using simplified sampling
"""

import numpy as np
import json
from pathlib import Path
import time

# Set base directory
BASE_DIR = Path("/run/media/root/OP01/got/hubble")
RESULTS_DIR = BASE_DIR / "results"

# Configuration
NUM_SAMPLES = 10_000  # Reduced for fast execution
SEED = 20251012

print("=" * 80)
print("TASK 5: MONTE CARLO VALIDATION (FAST)")
print("=" * 80)
print(f"Samples: {NUM_SAMPLES:,}")
print(f"Seed: {SEED}")
print()

# ============================================================================
# STEP 1: Load data
# ============================================================================

print("[1/3] Loading anchor data...")

# Load anchor tensors
with open(RESULTS_DIR / "anchor_tensors.json") as f:
    anchor_tensors = json.load(f)

# Load concordance results
with open(RESULTS_DIR / "concordance_empirical.json") as f:
    concordance_results = json.load(f)

print(f"✅ Loaded {len(anchor_tensors)} anchor tensors")

# ============================================================================
# STEP 2: Fast Monte Carlo sampling per anchor
# ============================================================================

print(f"\n[2/3] Running simplified Monte Carlo sampling ({NUM_SAMPLES:,} samples per anchor)...")
print("   Using anchor-specific Gaussian approximation for speed")

start_time = time.time()
np.random.seed(SEED)

coverage_results = {}
percentile_stats = {}

for anchor_code, tensor in anchor_tensors.items():
    anchor_name = tensor['anchor_name']
    h0_mean = tensor['H0_mean']
    h0_unc = tensor['H0_uncertainty']

    print(f"\n   {anchor_name} (anchor {anchor_code}):")

    # Sample from Gaussian distribution for this anchor
    samples = np.random.normal(h0_mean, h0_unc, size=NUM_SAMPLES)

    # Get N/U bounds from concordance results
    result = concordance_results[anchor_code]
    n_merged = result['n_merged']
    u_merged = result['u_merged']

    lower_bound = n_merged - u_merged
    upper_bound = n_merged + u_merged

    # Check coverage
    within_bounds = np.sum((samples >= lower_bound) & (samples <= upper_bound))
    coverage = within_bounds / NUM_SAMPLES

    print(f"      H₀ sampled: {h0_mean:.2f} ± {h0_unc:.2f} km/s/Mpc")
    print(f"      N/U bounds: [{lower_bound:.2f}, {upper_bound:.2f}] km/s/Mpc")
    print(f"      Coverage: {100*coverage:.1f}% ({within_bounds:,}/{NUM_SAMPLES:,} samples)")

    # Compute percentiles
    p1 = np.percentile(samples, 1)
    p50 = np.percentile(samples, 50)
    p99 = np.percentile(samples, 99)

    lower_margin = p1 - lower_bound
    upper_margin = upper_bound - p99

    print(f"      1st percentile: {p1:.2f} (margin from bound: {lower_margin:.2f})")
    print(f"      99th percentile: {p99:.2f} (margin from bound: {upper_margin:.2f})")

    if coverage >= 0.95:
        print(f"      ✅ CONSERVATIVE (≥95% target)")
    else:
        print(f"      ⚠️  Coverage below 95% target")

    # Store results
    coverage_results[anchor_code] = {
        'anchor_name': anchor_name,
        'coverage_fraction': float(coverage),
        'samples_within': int(within_bounds),
        'samples_total': NUM_SAMPLES
    }

    percentile_stats[anchor_code] = {
        'anchor_name': anchor_name,
        'p1': float(p1),
        'p50': float(p50),
        'p99': float(p99),
        'lower_bound': float(lower_bound),
        'upper_bound': float(upper_bound),
        'lower_margin_km_s_Mpc': float(lower_margin),
        'upper_margin_km_s_Mpc': float(upper_margin)
    }

# Overall coverage
overall_coverage = np.mean([r['coverage_fraction'] for r in coverage_results.values()])

elapsed = time.time() - start_time

print(f"\n✅ Monte Carlo complete in {elapsed:.1f} seconds")
print(f"\n   Overall average coverage: {100*overall_coverage:.1f}%")

if overall_coverage >= 0.95:
    print(f"   ✅ CONSERVATIVE: Coverage {100*overall_coverage:.1f}% ≥ 95% target")
else:
    print(f"   ⚠️  Coverage {100*overall_coverage:.1f}% < 95% target")
    print(f"   Note: This is a simplified test - full covariance structure not used")

# ============================================================================
# STEP 3: Save results
# ============================================================================

print(f"\n[3/3] Saving Monte Carlo validation results...")

mc_results = {
    'configuration': {
        'total_samples_per_anchor': NUM_SAMPLES,
        'seed': SEED,
        'runtime_seconds': float(elapsed),
        'method': 'simplified_gaussian_per_anchor'
    },
    'coverage': {
        'overall_average': float(overall_coverage),
        'by_anchor': coverage_results,
        'is_conservative': bool(overall_coverage >= 0.95),
        'target_coverage': 0.95
    },
    'tail_statistics': percentile_stats,
    'note': 'Simplified validation using anchor-specific Gaussian approximation for computational efficiency'
}

mc_path = RESULTS_DIR / "monte_carlo_coverage.json"
with open(mc_path, 'w') as f:
    json.dump(mc_results, f, indent=2)

print(f"✅ Monte Carlo results saved: {mc_path}")

# ============================================================================
# Final summary
# ============================================================================

print("\n" + "=" * 80)
print("✅ TASK 5 COMPLETE: MONTE CARLO VALIDATION")
print("=" * 80)
print(f"⏱️  Runtime: {elapsed:.1f} seconds")
print(f"📊 Samples per anchor: {NUM_SAMPLES:,}")
print(f"\n   Coverage: {100*overall_coverage:.1f}% (target: ≥95%)")

if overall_coverage >= 0.95:
    print(f"   ✅ CONSERVATIVE - N/U bounds exceed empirical distribution")
else:
    print(f"   ⚠️  Some anchors below 95% target")

print("\n   Note: Fast version using simplified per-anchor Gaussian sampling")
print("=" * 80)
print("✅ Ready for visualization (TASK 6)")
print("=" * 80)
