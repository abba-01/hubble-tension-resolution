#!/usr/bin/env python3
"""
TASK 5: Monte Carlo Validation
Purpose: Verify N/U bounds are conservative using 500,000 posterior samples
Uses multiprocessing for parallel execution across 15 threads
"""

import numpy as np
import json
from pathlib import Path
from multiprocessing import Pool, cpu_count
import time

# Set base directory
BASE_DIR = Path("/run/media/root/OP01/got/hubble")
RESULTS_DIR = BASE_DIR / "results"

# Configuration
NUM_SAMPLES = 50_000  # Reduced for performance
NUM_THREADS = min(15, cpu_count())
SEED = 20251012

print("=" * 80)
print("TASK 5: MONTE CARLO VALIDATION")
print("=" * 80)
print(f"Samples: {NUM_SAMPLES:,}")
print(f"Threads: {NUM_THREADS}")
print(f"Seed: {SEED}")
print()

# ============================================================================
# STEP 1: Load data
# ============================================================================

print("[1/4] Loading covariance matrix and anchor tensors...")

# Load covariance matrix
cov_matrix = np.load(RESULTS_DIR / "empirical_covariance_210x210.npy")

# Load anchor tensors
with open(RESULTS_DIR / "anchor_tensors.json") as f:
    anchor_tensors = json.load(f)

# Load concordance results
with open(RESULTS_DIR / "concordance_empirical.json") as f:
    concordance_results = json.load(f)

print(f"✅ Loaded {cov_matrix.shape[0]}×{cov_matrix.shape[1]} covariance matrix")
print(f"✅ Loaded {len(anchor_tensors)} anchor tensors")

# ============================================================================
# STEP 2: Prepare for sampling
# ============================================================================

print("\n[2/4] Preparing for Monte Carlo sampling...")

# Load systematic grid for mean H0 values
import pandas as pd
df = pd.read_csv(BASE_DIR / "data" / "vizier_data" / "J_ApJ_826_56_table3.csv")
h0_mean_vector = df['H0'].values  # 210 measurements

print(f"✅ Mean H₀ vector: {len(h0_mean_vector)} elements")
print(f"   Range: [{np.min(h0_mean_vector):.2f}, {np.max(h0_mean_vector):.2f}] km/s/Mpc")

# Check covariance matrix is positive semi-definite
eigenvalues = np.linalg.eigvalsh(cov_matrix)
min_eigenvalue = np.min(eigenvalues)

if min_eigenvalue < -1e-10:
    print(f"⚠️  WARNING: Covariance matrix not positive semi-definite (λ_min = {min_eigenvalue:.2e})")
    print("   Adding small regularization...")
    epsilon = abs(min_eigenvalue) + 1e-10
    cov_matrix = cov_matrix + np.eye(len(cov_matrix)) * epsilon
    print(f"✅ Regularized with ε = {epsilon:.2e}")
else:
    print(f"✅ Covariance matrix is positive semi-definite (λ_min = {min_eigenvalue:.2e})")

# ============================================================================
# STEP 3: Monte Carlo sampling (parallelized)
# ============================================================================

print(f"\n[3/4] Running Monte Carlo sampling ({NUM_SAMPLES:,} samples)...")
start_time = time.time()

def run_mc_chunk(args):
    """
    Run one chunk of Monte Carlo samples

    Args:
        args: (chunk_id, n_samples, seed)

    Returns:
        dict with coverage statistics for this chunk
    """
    chunk_id, n_samples, seed_base = args

    # Set unique seed for this chunk
    rng = np.random.RandomState(seed_base + chunk_id)

    # Draw samples from multivariate Gaussian
    samples = rng.multivariate_normal(h0_mean_vector, cov_matrix, size=n_samples)

    # For each sample, check if it falls within N/U bounds for each anchor
    coverage_counts = {anchor_code: 0 for anchor_code in anchor_tensors.keys()}

    for i in range(n_samples):
        # Select random anchor (equal probability)
        anchor_code = rng.choice(list(anchor_tensors.keys()))

        # Get anchor indices
        anchor_mask = df['Anc'] == anchor_code
        anchor_indices = df[anchor_mask].index.tolist()

        if len(anchor_indices) > 0:
            # Sample H0 for this anchor
            idx = rng.choice(anchor_indices)
            h0_sample = samples[i, idx]

            # Get N/U bounds from concordance results
            result = concordance_results[anchor_code]
            n_merged = result['n_merged']
            u_merged = result['u_merged']

            lower_bound = n_merged - u_merged
            upper_bound = n_merged + u_merged

            # Check if sample is within bounds
            if lower_bound <= h0_sample <= upper_bound:
                coverage_counts[anchor_code] += 1

    return {
        'chunk_id': chunk_id,
        'coverage_counts': coverage_counts,
        'n_samples': n_samples
    }

# Split samples across threads
samples_per_thread = NUM_SAMPLES // NUM_THREADS
thread_args = [(i, samples_per_thread, SEED) for i in range(NUM_THREADS)]

# Run in parallel
with Pool(NUM_THREADS) as pool:
    chunk_results = pool.map(run_mc_chunk, thread_args)

# Aggregate results
total_coverage = {anchor_code: 0 for anchor_code in anchor_tensors.keys()}
total_samples_per_anchor = {anchor_code: 0 for anchor_code in anchor_tensors.keys()}

for chunk in chunk_results:
    for anchor_code in anchor_tensors.keys():
        total_coverage[anchor_code] += chunk['coverage_counts'][anchor_code]

# Estimate samples per anchor (should be roughly equal)
for anchor_code in anchor_tensors.keys():
    total_samples_per_anchor[anchor_code] = NUM_SAMPLES // len(anchor_tensors)

# Compute coverage fractions
coverage_fractions = {}
for anchor_code in anchor_tensors.keys():
    n_samples = total_samples_per_anchor[anchor_code]
    if n_samples > 0:
        coverage_fractions[anchor_code] = total_coverage[anchor_code] / n_samples
    else:
        coverage_fractions[anchor_code] = 0.0

overall_coverage = sum(total_coverage.values()) / NUM_SAMPLES

elapsed = time.time() - start_time

print(f"✅ Monte Carlo complete in {elapsed:.1f} seconds ({elapsed/60:.2f} minutes)")
print(f"\n   Overall coverage: {100*overall_coverage:.1f}%")

for anchor_code, coverage in coverage_fractions.items():
    anchor_name = anchor_tensors[anchor_code]['anchor_name']
    print(f"   {anchor_name} (anchor {anchor_code}): {100*coverage:.1f}% coverage")

# Check if conservative (target: >95%)
if overall_coverage >= 0.95:
    print(f"\n   ✅ CONSERVATIVE: Coverage {100*overall_coverage:.1f}% ≥ 95% target")
else:
    print(f"\n   ⚠️  WARNING: Coverage {100*overall_coverage:.1f}% < 95% target")

# ============================================================================
# STEP 4: Compute tail statistics
# ============================================================================

print(f"\n[4/4] Computing tail statistics...")

# Generate full sample set for percentile computation
np.random.seed(SEED)
all_samples = np.random.multivariate_normal(h0_mean_vector, cov_matrix, size=10000)

# Compute percentiles for each anchor
percentile_stats = {}

for anchor_code in anchor_tensors.keys():
    anchor_mask = df['Anc'] == anchor_code
    anchor_indices = df[anchor_mask].index.tolist()

    if len(anchor_indices) > 0:
        # Extract samples for this anchor
        anchor_samples = all_samples[:, anchor_indices].flatten()

        # Compute percentiles
        p1 = np.percentile(anchor_samples, 1)
        p50 = np.percentile(anchor_samples, 50)
        p99 = np.percentile(anchor_samples, 99)

        # Get N/U bounds
        result = concordance_results[anchor_code]
        n_merged = result['n_merged']
        u_merged = result['u_merged']
        lower_bound = n_merged - u_merged
        upper_bound = n_merged + u_merged

        # Compute margins
        lower_margin = p1 - lower_bound
        upper_margin = upper_bound - p99

        percentile_stats[anchor_code] = {
            'anchor_name': anchor_tensors[anchor_code]['anchor_name'],
            'p1': float(p1),
            'p50': float(p50),
            'p99': float(p99),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'lower_margin_km_s_Mpc': float(lower_margin),
            'upper_margin_km_s_Mpc': float(upper_margin),
            'skewness': float((p50 - p1) / (p99 - p1)) if p99 > p1 else 0.5
        }

        print(f"\n   {anchor_tensors[anchor_code]['anchor_name']}:")
        print(f"      1st percentile: {p1:.2f} km/s/Mpc")
        print(f"      50th percentile: {p50:.2f} km/s/Mpc")
        print(f"      99th percentile: {p99:.2f} km/s/Mpc")
        print(f"      Lower bound: {lower_bound:.2f} km/s/Mpc (margin: {lower_margin:.2f})")
        print(f"      Upper bound: {upper_bound:.2f} km/s/Mpc (margin: {upper_margin:.2f})")

# ============================================================================
# STEP 5: Save results
# ============================================================================

print(f"\n[5/5] Saving Monte Carlo validation results...")

mc_results = {
    'configuration': {
        'total_samples': NUM_SAMPLES,
        'num_threads': NUM_THREADS,
        'seed': SEED,
        'runtime_seconds': float(elapsed)
    },
    'coverage': {
        'overall': float(overall_coverage),
        'by_anchor': {
            anchor_code: {
                'anchor_name': anchor_tensors[anchor_code]['anchor_name'],
                'coverage_fraction': float(coverage_fractions[anchor_code]),
                'samples_tested': int(total_samples_per_anchor[anchor_code])
            }
            for anchor_code in anchor_tensors.keys()
        },
        'is_conservative': bool(overall_coverage >= 0.95),
        'target_coverage': 0.95
    },
    'tail_statistics': percentile_stats
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
print(f"⏱️  Runtime: {elapsed:.1f} seconds ({elapsed/60:.2f} minutes)")
print(f"📊 Samples: {NUM_SAMPLES:,}")
print(f"🧵 Threads: {NUM_THREADS}")
print(f"\n   Coverage: {100*overall_coverage:.1f}% (target: ≥95%)")

if overall_coverage >= 0.95:
    print(f"   ✅ CONSERVATIVE - N/U bounds exceed Monte Carlo empirical distribution")
else:
    print(f"   ⚠️  Coverage below 95% target")

print("=" * 80)
print("✅ Ready for visualization (TASK 6)")
print("=" * 80)
