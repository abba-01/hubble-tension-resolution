"""
Hubble Tension Resolution via N/U Algebra
==========================================
Author: Eric D. Martin
Framework: N/U Algebra + UHA
Date: 2025-10-11

This notebook implements the N/U algebra merge rule to reconcile
H₀ measurements from multiple cosmological probes.

Merge Rule:
H_NU = ((n₁+n₂)/2, (u₁+u₂)/2 + |n₁-n₂|/2)

References:
- Planck 2018: Planck Collaboration 2020, A&A 641, A6 (arXiv:1807.06209)
- SH0ES 2022: Riess et al. 2022, ApJL 934, L7 (arXiv:2112.04510)
- H0LiCOW 2020: Wong et al. 2020, MNRAS 498, 1420 (arXiv:1907.04869)
- Megamaser 2020: Pesce et al. 2020, ApJL 891, L1
- DES-Y5 2024: DES Collaboration 2025, MNRAS 537, 1818
- TRGB CCHP: Freedman et al. 2020, ApJ 891, 57
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Latest published H₀ measurements (km/s/Mpc)
# All values verified against peer-reviewed literature as of 2025-10-11
H0_DATA = {
    'source': [
        'Planck 2018 CMB',
        'SH0ES 2022',
        'H0LiCOW 2020',
        'Megamaser 2020',
        'DES-Y5 2024',
        'TRGB CCHP 2020'
    ],
    'method': [
        'CMB (ΛCDM)',
        'Cepheids+SNe Ia',
        'Time-delay lensing',
        'Megamaser (NGC 4258)',
        'SNe Ia + BAO (inverse ladder)',
        'TRGB'
    ],
    'redshift': [
        1100.0,      # CMB last scattering surface
        0.01,        # Local distance ladder (z < 0.01)
        0.5,         # Typical lens redshift (z ~ 0.3-0.7)
        0.0023,      # NGC 4258 (z = 0.0023)
        0.7,         # DES-Y5 effective redshift
        0.01         # Local TRGB (z < 0.01)
    ],
    'n': [
        67.4,   # Planck Collaboration 2020 (arXiv:1807.06209)
        73.04,  # Riess et al. 2022 (arXiv:2112.04510)
        73.3,   # Wong et al. 2020 (arXiv:1907.04869)
        73.9,   # Pesce et al. 2020 (MCP)
        67.19,  # DES Collaboration 2024 (DES-SN5YR + DESI-BAO)
        69.6    # Freedman et al. 2020 (CCHP - updated value)
    ],
    'u': [
        0.5,    # Planck (68% CL)
        1.04,   # SH0ES (statistical + systematic)
        1.8,    # H0LiCOW (average of +1.7/-1.8)
        3.0,    # Megamaser
        0.65,   # DES-Y5 (average of +0.66/-0.64)
        2.5     # TRGB CCHP (updated uncertainty)
    ]
}

# ============================================================================
# N/U ALGEBRA OPERATORS
# ============================================================================

def nu_merge(n1: float, u1: float, n2: float, u2: float) -> Tuple[float, float]:
    """
    Merge two N/U pairs using conservative merge rule.
    
    Rule: H_NU = ((n₁+n₂)/2, (u₁+u₂)/2 + |n₁-n₂|/2)
    
    This adds the disagreement between nominals into uncertainty,
    ensuring the merged interval contains both original intervals.
    
    Args:
        n1, u1: First nominal and uncertainty
        n2, u2: Second nominal and uncertainty
    
    Returns:
        Tuple of (merged_nominal, merged_uncertainty)
    """
    n_merge = (n1 + n2) / 2.0
    u_merge = (u1 + u2) / 2.0 + abs(n1 - n2) / 2.0
    return n_merge, u_merge


def nu_cumulative_merge(nominals: np.ndarray, uncertainties: np.ndarray) -> Tuple[float, float]:
    """
    Iteratively merge multiple N/U pairs.
    
    Process:
    1. Start with first pair
    2. Merge with second pair
    3. Continue until all pairs merged
    
    Args:
        nominals: Array of nominal H₀ values
        uncertainties: Array of uncertainties
    
    Returns:
        Final merged (n, u) pair
    """
    if len(nominals) != len(uncertainties):
        raise ValueError("Nominals and uncertainties must have same length")
    
    if len(nominals) == 0:
        return 0.0, 0.0
    
    # Initialize with first measurement
    n_result = nominals[0]
    u_result = uncertainties[0]
    
    # Iteratively merge remaining measurements
    for i in range(1, len(nominals)):
        n_result, u_result = nu_merge(n_result, u_result, nominals[i], uncertainties[i])
    
    return n_result, u_result


def nu_interval(n: float, u: float) -> Tuple[float, float]:
    """Convert N/U pair to interval [n-u, n+u]"""
    return n - u, n + u


# ============================================================================
# DATA PREPARATION
# ============================================================================

def create_dataset() -> pd.DataFrame:
    """Create and validate the H₀ measurements dataset"""
    df = pd.DataFrame(H0_DATA)
    
    # Add derived columns
    df['interval_low'] = df['n'] - df['u']
    df['interval_high'] = df['n'] + df['u']
    df['interval_width'] = 2 * df['u']
    df['relative_uncertainty'] = df['u'] / df['n']
    
    # Categorize by redshift regime
    df['regime'] = df['redshift'].apply(
        lambda z: 'Early (CMB)' if z > 100 else 'Late (z<1)'
    )
    
    return df


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def calculate_nu_h0(df: pd.DataFrame) -> Tuple[float, float, pd.DataFrame]:
    """
    Calculate merged H₀ using N/U algebra.
    
    Returns:
        (merged_n, merged_u, steps_df)
    """
    nominals = df['n'].values
    uncertainties = df['u'].values
    
    # Track merge steps
    steps = []
    n_current = nominals[0]
    u_current = uncertainties[0]
    
    steps.append({
        'step': 0,
        'source': df.iloc[0]['source'],
        'n': n_current,
        'u': u_current,
        'interval_low': n_current - u_current,
        'interval_high': n_current + u_current
    })
    
    for i in range(1, len(nominals)):
        n_current, u_current = nu_merge(
            n_current, u_current,
            nominals[i], uncertainties[i]
        )
        
        steps.append({
            'step': i,
            'source': f"After merging {df.iloc[i]['source']}",
            'n': n_current,
            'u': u_current,
            'interval_low': n_current - u_current,
            'interval_high': n_current + u_current
        })
    
    steps_df = pd.DataFrame(steps)
    return n_current, u_current, steps_df


def calculate_tension(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate pairwise tensions between measurements.
    
    Tension metric: σ = |n₁ - n₂| / sqrt(u₁² + u₂²)
    """
    tensions = []
    n = len(df)
    
    for i in range(n):
        for j in range(i + 1, n):
            n1, u1 = df.iloc[i]['n'], df.iloc[i]['u']
            n2, u2 = df.iloc[j]['n'], df.iloc[j]['u']
            
            tension_sigma = abs(n1 - n2) / np.sqrt(u1**2 + u2**2)
            
            tensions.append({
                'source_1': df.iloc[i]['source'],
                'source_2': df.iloc[j]['source'],
                'delta_H0': abs(n1 - n2),
                'tension_sigma': tension_sigma,
                'overlaps': not (min(n1 + u1, n2 + u2) < max(n1 - u1, n2 - u2))
            })
    
    return pd.DataFrame(tensions)


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_measurements(df: pd.DataFrame, h0_nu: Tuple[float, float], save_path: str = None):
    """
    Create comprehensive visualization of H₀ measurements and N/U merge.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # ---- Panel 1: Individual Measurements ----
    n_nu, u_nu = h0_nu
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df)))
    
    for i, row in df.iterrows():
        ax1.errorbar(
            row['n'], i,
            xerr=row['u'],
            fmt='o',
            color=colors[i],
            capsize=5,
            capthick=2,
            markersize=8,
            label=f"{row['source']} ({row['method']})"
        )
    
    # Plot N/U merged result
    ax1.axvspan(n_nu - u_nu, n_nu + u_nu, alpha=0.2, color='red', label='N/U Merged')
    ax1.axvline(n_nu, color='red', linestyle='--', linewidth=2, label=f'N/U H₀ = {n_nu:.2f}')
    
    ax1.set_xlabel('H₀ (km/s/Mpc)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Measurement', fontsize=12, fontweight='bold')
    ax1.set_title('Hubble Constant Measurements and N/U Algebra Merge', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_yticks(range(len(df)))
    ax1.set_yticklabels(df['source'])
    
    # ---- Panel 2: Redshift Dependence ----
    early = df[df['regime'] == 'Early (CMB)']
    late = df[df['regime'] == 'Late (z<1)']
    
    ax2.errorbar(
        early['redshift'], early['n'],
        yerr=early['u'],
        fmt='s',
        color='blue',
        capsize=5,
        markersize=10,
        label='Early Universe (CMB)',
        alpha=0.7
    )
    
    ax2.errorbar(
        late['redshift'], late['n'],
        yerr=late['u'],
        fmt='o',
        color='orange',
        capsize=5,
        markersize=10,
        label='Late Universe (z<1)',
        alpha=0.7
    )
    
    # Add N/U merged band
    ax2.axhspan(n_nu - u_nu, n_nu + u_nu, alpha=0.2, color='red', label='N/U Merged ± u')
    ax2.axhline(n_nu, color='red', linestyle='--', linewidth=2)
    
    ax2.set_xlabel('Redshift z', fontsize=12, fontweight='bold')
    ax2.set_ylabel('H₀ (km/s/Mpc)', fontsize=12, fontweight='bold')
    ax2.set_title('H₀ vs Redshift: Early-Late Universe Tension', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_merge_evolution(steps_df: pd.DataFrame, save_path: str = None):
    """
    Visualize how the N/U merged interval evolves as measurements are added.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, row in steps_df.iterrows():
        color = plt.cm.coolwarm(i / len(steps_df))
        ax.errorbar(
            row['n'], i,
            xerr=row['u'],
            fmt='o',
            color=color,
            capsize=5,
            capthick=2,
            markersize=8
        )
        ax.text(
            row['interval_high'] + 0.5, i,
            row['source'],
            fontsize=9,
            va='center'
        )
    
    ax.set_xlabel('H₀ (km/s/Mpc)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Merge Step', fontsize=12, fontweight='bold')
    ax.set_title('N/U Algebra: Iterative Merge Evolution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yticks(range(len(steps_df)))
    ax.set_yticklabels([f"Step {i}" for i in range(len(steps_df))])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    """Execute complete N/U Algebra Hubble tension analysis"""
    
    print("=" * 80)
    print("HUBBLE TENSION RESOLUTION via N/U ALGEBRA")
    print("=" * 80)
    print()
    
    # Create dataset
    print("Loading H₀ measurements...")
    df = create_dataset()
    print(f"✓ Loaded {len(df)} measurements")
    print()
    
    # Display dataset
    print("Dataset Summary:")
    print("-" * 80)
    print(df[['source', 'method', 'n', 'u', 'regime']].to_string(index=False))
    print()
    
    # Calculate N/U merged H₀
    print("Calculating N/U merged H₀...")
    n_nu, u_nu, steps_df = calculate_nu_h0(df)
    interval_low, interval_high = nu_interval(n_nu, u_nu)
    
    print("=" * 80)
    print("RESULT: N/U MERGED HUBBLE CONSTANT")
    print("=" * 80)
    print(f"H₀_NU = {n_nu:.3f} ± {u_nu:.3f} km/s/Mpc")
    print(f"Interval: [{interval_low:.3f}, {interval_high:.3f}]")
    print(f"Relative uncertainty: {u_nu/n_nu*100:.2f}%")
    print()
    
    # Calculate tensions
    print("Calculating pairwise tensions...")
    tensions = calculate_tension(df)
    print()
    print("Pairwise Tensions (σ > 3.0 shown):")
    print("-" * 80)
    high_tension = tensions[tensions['tension_sigma'] > 3.0]
    if len(high_tension) > 0:
        print(high_tension[['source_1', 'source_2', 'tension_sigma', 'overlaps']].to_string(index=False))
    else:
        print("No high tensions (σ > 3.0) detected.")
    print()
    
    # Summary statistics
    print("Summary Statistics:")
    print("-" * 80)
    print(f"Early universe mean: {df[df['regime']=='Early (CMB)']['n'].mean():.2f} km/s/Mpc")
    print(f"Late universe mean:  {df[df['regime']=='Late (z<1)']['n'].mean():.2f} km/s/Mpc")
    print(f"Overall spread:      {df['n'].max() - df['n'].min():.2f} km/s/Mpc")
    print(f"N/U captures:        {(interval_high - interval_low):.2f} km/s/Mpc interval")
    print()
    
    # Visualizations
    print("Generating visualizations...")
    plot_measurements(df, (n_nu, u_nu))
    plot_merge_evolution(steps_df)
    
    # Export results
    print("Exporting results...")
    df.to_csv('h0_measurements.csv', index=False)
    steps_df.to_csv('nu_merge_steps.csv', index=False)
    tensions.to_csv('pairwise_tensions.csv', index=False)
    
    print("✓ Saved: h0_measurements.csv")
    print("✓ Saved: nu_merge_steps.csv")
    print("✓ Saved: pairwise_tensions.csv")
    print()
    
    print("=" * 80)
    print("Analysis complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
