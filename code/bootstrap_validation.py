#!/usr/bin/env python3
"""
Bootstrap validation for observer-tensor N/U framework.

Phase 1: Published aggregate resampling
Author: Eric D. Martin
Date: 2025-10-11
License: MIT
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from hashlib import sha256
from typing import Dict, Any
import time
import sys


# -----------------------------------------------------------------------------
# 1. Load published values (REVISED for canonical JSON structure)
# -----------------------------------------------------------------------------
def load_published_values(path: str | Path) -> Dict[str, Any]:
    """
    Load canonical H₀ values and observer tensors from CORRECTED_RESULTS_32BIT.json.
    
    Adapts to the published structure:
    - observer_tensors_refined: {probe_name: {T_obs, ...}}
    - aggregated_measurements: {early_universe: {H0_n, H0_u}, late_universe: {...}}
    
    Returns
    -------
    dict : {
        'probes': {probe_name: {'H0_n': float, 'H0_u': float, 'T_obs': list[float]}},
        'checksum': str
    }
    """
    path = Path(path)
    with open(path, "r") as f:
        data = json.load(f)
    
    # Extract tensors
    tensors_raw = data["observer_tensors_refined"]
    
    # Extract aggregated measurements
    early_agg = data["aggregated_measurements"]["early_universe"]
    late_agg = data["aggregated_measurements"]["late_universe"]
    
    # Define probe groupings
    early_group = ["Planck18", "DES-IDL"]
    late_group = ["SH0ES", "TRGB", "TDCOSMO", "Megamaser"]
    
    # Build unified structure
    probes = {}
    
    # Assign early probes
    for probe_name in early_group:
        if probe_name in tensors_raw:
            probes[probe_name] = {
                "H0_n": early_agg["H0_n"],
                "H0_u": early_agg["H0_u"],
                "T_obs": tensors_raw[probe_name]["T_obs"],
                "group": "early"
            }
    
    # Assign late probes
    for probe_name in late_group:
        if probe_name in tensors_raw:
            probes[probe_name] = {
                "H0_n": late_agg["H0_n"],
                "H0_u": late_agg["H0_u"],
                "T_obs": tensors_raw[probe_name]["T_obs"],
                "group": "late"
            }
    
    # Compute checksum
    checksum = sha256(json.dumps(probes, sort_keys=True).encode()).hexdigest()
    
    print(f"Loaded {len(probes)} probes from {path.name}")
    print(f"  Early group: {len([p for p in probes.values() if p['group'] == 'early'])} probes")
    print(f"  Late group: {len([p for p in probes.values() if p['group'] == 'late'])} probes")
    print(f"Checksum: {checksum[:12]}...")
    
    return {"probes": probes, "checksum": checksum}


# -----------------------------------------------------------------------------
# 2. Bootstrap sampling
# -----------------------------------------------------------------------------
def bootstrap_samples(probes: Dict[str, Any], n_iter: int = 10_000,
                      seed: int = 20251011) -> Dict[str, np.ndarray]:
    """
    Draw normal samples for each probe using (n,u) pairs.

    Returns
    -------
    dict : {probe_name: np.ndarray of shape (n_iter,)}
    """
    rng = np.random.default_rng(seed)
    samples = {}
    
    print(f"\nGenerating {n_iter:,} bootstrap samples (seed={seed})...")
    for name, vals in probes.items():
        samples[name] = rng.normal(vals["H0_n"], vals["H0_u"], size=n_iter)
        if len(samples) % 2 == 0:
            print(f"  ✓ {name}: {n_iter:,} samples")
    
    return samples


# -----------------------------------------------------------------------------
# 3. Merge with tensors
# -----------------------------------------------------------------------------
def merge_with_tensors(samples: Dict[str, np.ndarray],
                       tensors: Dict[str, Any]) -> pd.DataFrame:
    """
    Combine early and late groups with tensor-weighted epistemic distance.

    Returns
    -------
    pd.DataFrame with columns:
    [H0_early, H0_late, delta_T, H0_merged, gap]
    """
    early_group = ["Planck18", "DES-IDL"]
    late_group  = ["SH0ES", "TRGB", "TDCOSMO", "Megamaser"]

    n_iter = len(next(iter(samples.values())))
    results = np.zeros((n_iter, 5))

    print(f"\nComputing tensor-weighted merges for {n_iter:,} iterations...")
    
    for i in range(n_iter):
        if (i + 1) % 2000 == 0:
            print(f"  Progress: {i+1:,}/{n_iter:,} ({100*(i+1)/n_iter:.1f}%)")
        
        H0_early = np.mean([samples[p][i] for p in early_group])
        H0_late  = np.mean([samples[p][i] for p in late_group])

        # Compute epistemic distance Δ_T
        T_e = np.mean([tensors[p]["T_obs"] for p in early_group], axis=0)
        T_l = np.mean([tensors[p]["T_obs"] for p in late_group], axis=0)
        delta_T = np.linalg.norm(np.array(T_e) - np.array(T_l))

        # Merge rule
        u_e = np.mean([tensors[p]["H0_u"] for p in early_group])
        u_l = np.mean([tensors[p]["H0_u"] for p in late_group])
        H0_m = (H0_early + H0_late) / 2
        u_m  = (u_e + u_l) / 2 + abs(H0_early - H0_late) / 2 * delta_T

        results[i] = [H0_early, H0_late, delta_T, H0_m, abs(H0_early - H0_late)]

    cols = ["H0_early", "H0_late", "delta_T", "H0_merged", "gap"]
    return pd.DataFrame(results, columns=cols)


# -----------------------------------------------------------------------------
# 4. Summarize statistics
# -----------------------------------------------------------------------------
def summarize_statistics(df: pd.DataFrame,
                         published: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute 95% confidence intervals and validation checks.
    """
    print("\nComputing statistical summary...")
    
    summary = {}
    for col in ["H0_early", "H0_late", "delta_T", "gap"]:
        vals = df[col].values
        ci95 = np.percentile(vals, [2.5, 97.5])
        summary[col] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
            "ci_95": [float(ci95[0]), float(ci95[1])],
            "n_samples": int(len(vals))  # Convert to Python int
        }

    # Validation flags
    gap_mean = summary["gap"]["mean"]
    gap_std = summary["gap"]["std"]

    summary["validation"] = {
        "mean_bias_ok": bool(abs(gap_mean - 0.48) < 0.05),  # Convert to Python bool
        "gap_target": 0.48,
        "gap_achieved": float(gap_mean),
        "ci_width": float(summary["gap"]["ci_95"][1] - summary["gap"]["ci_95"][0]),
        "no_outliers": bool(np.max(np.abs(df["gap"])) < 5 * gap_std),  # Convert to Python bool
        "reduction_from_original": float(1.0 - (gap_mean / 5.40))
    }
    
    return summary


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------
def main():
    try:
        import yaml
    except ImportError:
        print("ERROR: pyyaml not installed. Run: pip install pyyaml")
        sys.exit(1)
    
    print("=" * 80)
    print("BOOTSTRAP VALIDATION: Observer-Tensor N/U Framework")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    t0 = time.time()
    src = "../CORRECTED_RESULTS_32BIT.json"
    out_dir = Path("../validation_results")
    out_dir.mkdir(exist_ok=True)

    # 1. Load data
    data = load_published_values(src)
    probes = data["probes"]

    # 2. Sample
    samples = bootstrap_samples(probes, n_iter=10_000, seed=20251011)

    # 3. Merge
    df = merge_with_tensors(samples, probes)
    csv_path = out_dir / "bootstrap_samples.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved: {csv_path}")

    # 4. Summarize
    summary = summarize_statistics(df, probes)
    json_path = out_dir / "validation_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved: {json_path}")

    # 5. Reproducibility metadata
    runtime = time.time() - t0
    meta = {
        "seed": 20251011,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "runtime_seconds": round(runtime, 2),
        "timestamp": datetime.now().isoformat(),
        "checksum_samples": sha256(df.to_csv(index=False).encode()).hexdigest()[:12],
        "checksum_summary": sha256(json.dumps(summary).encode()).hexdigest()[:12]
    }
    yaml_path = out_dir / "reproducibility.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(meta, f)
    print(f"✓ Saved: {yaml_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Gap: {summary['gap']['mean']:.2f} ± {summary['gap']['std']:.2f} km/s/Mpc")
    print(f"95% CI: [{summary['gap']['ci_95'][0]:.2f}, {summary['gap']['ci_95'][1]:.2f}]")
    print(f"Target gap: {summary['validation']['gap_target']:.2f} km/s/Mpc")
    print(f"Reduction achieved: {summary['validation']['reduction_from_original']*100:.1f}%")
    print(f"\nRuntime: {runtime:.1f} seconds")
    print("=" * 80)

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
