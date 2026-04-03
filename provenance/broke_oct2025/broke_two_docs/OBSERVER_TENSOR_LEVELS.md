# Observer Tensor Application: Three Levels

## Visual Comparison of Observer Tensor Application Strategies

### LEVEL 1: Anchor-Level (CURRENT APPROACH) ✅

```
VizieR Systematic Grid (210 H₀ configs)
│
├─ NGC4258 subset (70 configs)     ─→  Sample 5,000 × → [T_N] → w_N = 33.5%
│  └─ μ = 72.52 ± 2.39                 Observer tensor
│
├─ Milky Way subset (70 configs)   ─→  Sample 5,000 × → [T_M] → w_M = 32.9%
│  └─ μ = 76.17 ± 2.34                 Observer tensor
│
└─ LMC subset (70 configs)         ─→  Sample 5,000 × → [T_L] → w_L = 33.6%
   └─ μ = 72.27 ± 2.62                 Observer tensor
                                                    ↓
                                        Weighted aggregate
                                                    ↓
                                    SH0ES_weighted = 73.64 ± 3.03
```

**Interpretation:**
- 3 observer tensors (one per anchor)
- Each anchor is an epistemic "observer"
- Weight by inverse epistemic distance: w = 1/(1 + |T|)
- Final weighted mean across 3 anchors

**Pros:**
- Simple and interpretable
- Natural epistemic grouping (different geometric methods)
- Computationally efficient
- Clear physical meaning

**Cons:**
- Loses granularity of systematic variations
- All configs within anchor treated equally
- Cannot distinguish PL relation effects from reddening effects

---

### LEVEL 2: Configuration-Level (ALTERNATIVE)

```
VizieR Systematic Grid (210 H₀ configs)
│
├─ Config 1: N,P1,4.5,S1 → H₀=73.24±1.62 → [T_001] → w_001
├─ Config 2: N,P1,4.5,S2 → H₀=72.89±1.58 → [T_002] → w_002
├─ Config 3: N,P2,4.0,S1 → H₀=74.11±1.71 → [T_003] → w_003
│  ...
└─ Config 210: L,P3,4.5,S3 → H₀=71.43±2.03 → [T_210] → w_210
                                             ↓
                                  Weighted aggregate
                                             ↓
                              SH0ES_weighted = ???
```

**Interpretation:**
- 210 observer tensors (one per config)
- Each systematic choice is an epistemic "observer"
- Weight by inverse epistemic distance
- Final weighted mean across 210 configs

**How to implement:**
```python
for i in range(210):
    config = systematic_grid[i]

    # Compute observer tensor for this config
    T_i = compute_observer_tensor(
        anchor=config['Anc'],
        PL_relation=config['PL'],
        breakpoint=config['break'],
        selection=config['sel'],
        reddening=config['reddening'],
        metallicity=config['metallicity']
    )

    # Weight by inverse epistemic distance
    w_i = 1.0 / (1.0 + tensor_magnitude(T_i))

    # Add to weighted sum
    weighted_sum += w_i * config['H0']
    total_weight += w_i

SH0ES_weighted = weighted_sum / total_weight
```

**Pros:**
- Finer granularity
- Can distinguish different systematic effects
- More nuanced epistemic weighting
- Captures methodology differences within anchors

**Cons:**
- Need to define observer tensors for each config
- What is P_m for "PL relation choice"?
- What is 0_t for "reddening law choice"?
- Computationally more complex
- May be over-fitting

---

### LEVEL 3: SN-Level (MOST GRANULAR)

```
Pantheon+SH0ES.dat (2,287 individual SNe)
│
For each SN:
├─ SN1 × Config1 → μ → H₀ → [T_SN1_C1] → w
├─ SN1 × Config2 → μ → H₀ → [T_SN1_C2] → w
│  ...
├─ SN1 × Config210 → μ → H₀ → [T_SN1_C210] → w
├─ SN2 × Config1 → μ → H₀ → [T_SN2_C1] → w
│  ...
└─ SN2287 × Config210 → μ → H₀ → [T_SN2287_C210] → w
                                    ↓
                          Weighted aggregate
                                    ↓
                       SH0ES_weighted = ???

Total: 2,287 × 210 = 480,270 observer tensors!
```

**Interpretation:**
- Each (SN, systematic config) pair is an epistemic "observer"
- Full granularity of measurement process
- Weight each observation by epistemic distance

**How to implement:**
```python
for sn in all_supernovae:  # 2,287 SNe
    for config in systematic_configs:  # 210 configs
        # Compute distance modulus for this SN with this config
        mu = compute_distance_modulus(
            sn,
            PL=config['PL'],
            reddening=config['reddening'],
            etc.
        )

        # Convert to H₀
        H0 = distance_modulus_to_H0(mu, sn.redshift)

        # Compute observer tensor for (SN, config)
        T = compute_observer_tensor(
            sn=sn,
            config=config,
            methodology=config['anchor']
        )

        # Weight by inverse epistemic distance
        w = 1.0 / (1.0 + tensor_magnitude(T))

        weighted_sum += w * H0
        total_weight += w

SH0ES_weighted = weighted_sum / total_weight
```

**Pros:**
- Maximum granularity
- Full transparency from SNe → H₀
- Can weight each observation individually
- Most faithful to observer tensor philosophy

**Cons:**
- Extremely computationally expensive
- 480,270 observer tensors to compute
- Need full analysis pipeline (not just results)
- How do you define observer tensor for (SN, config)?
- Likely to overfit or be dominated by noise
- Diminishing returns vs Level 2

---

## Comparison Table

| Aspect | Level 1: Anchor | Level 2: Config | Level 3: SN |
|--------|----------------|-----------------|-------------|
| **# Observer tensors** | 3 | 210 | 480,270 |
| **Granularity** | Coarse | Medium | Fine |
| **Interpretability** | High | Medium | Low |
| **Computation** | Fast | Moderate | Slow |
| **Data required** | Grid means | Grid full | Raw SNe |
| **Physical meaning** | Clear | Moderate | Unclear |
| **Risk of overfitting** | Low | Medium | High |
| **Current implementation** | ✅ YES | ❌ NO | ❌ NO |

---

## Detailed Example: Level 1 (Current)

### Input: 3 Anchor Posteriors

```
NGC4258:  72.52 ± 2.39 km/s/Mpc  (5,000 samples)
MilkyWay: 76.17 ± 2.34 km/s/Mpc  (5,000 samples)
LMC:      72.27 ± 2.62 km/s/Mpc  (5,000 samples)
```

### Observer Tensors (Anchor-Level)

```
T_NGC4258 = [a_N, P_m_N, 0_t_N, 0_a_N]
          = [0.9670, 0.05, -0.02, 0.5]  # Distance ladder, geometric
          |T| = 0.5183

T_MilkyWay = [a_M, P_m_M, 0_t_M, 0_a_M]
           = [0.9693, 0.05, -0.02, 0.5]  # Distance ladder, geometric
           |T| = 0.5448

T_LMC = [a_L, P_m_L, 0_t_L, 0_a_L]
      = [0.9638, 0.05, -0.02, 0.5]  # Distance ladder, geometric
      |T| = 0.5169
```

Note: Observer tensors are very similar because all use distance ladder.
Small differences in 'a' component due to measurement precision.

### Weights

```
w_NGC4258 = 1 / (1 + 0.5183) = 0.6589  →  33.5% (normalized)
w_MilkyWay = 1 / (1 + 0.5448) = 0.6473  →  32.9% (normalized)
w_LMC = 1 / (1 + 0.5169) = 0.6594  →  33.6% (normalized)
```

### Weighted Mean

```
μ_weighted = 0.335 × 72.52 + 0.329 × 76.17 + 0.336 × 72.27
           = 24.29 + 25.06 + 24.28
           = 73.63 ≈ 73.64 km/s/Mpc
```

### Weighted Uncertainty

```
For each sample i:
  weighted_sample_i = w_anchor × sample_i

σ_weighted = std(all_weighted_samples)
           = 3.03 km/s/Mpc
```

This is the **anchor spread** plus **intra-anchor variation**.

---

## Hypothetical Example: Level 2 (Config-Level)

### Input: 210 Configuration H₀ Values

```
Config   Anc  PL  break  sel   H₀       unc   → Observer Tensor → Weight
------   ---  --  -----  ---   ------   ----     ---------------   ------
   1      N   P1   4.5    S1   73.24   1.62   [a,P_m,0_t,0_a,θ_PL,θ_red]  w_1
   2      N   P1   4.5    S2   72.89   1.58   [...]                        w_2
   3      N   P2   4.0    S1   74.11   1.71   [...]                        w_3
  ...
 210      L   P3   4.5    S3   71.43   2.03   [...]                        w_210
```

### How to Define Config-Level Observer Tensors?

Need to extend observer tensor to include systematic choices:

```
T_config = [a, P_m, 0_t, 0_a, θ_PL, θ_red, θ_metal, ...]

where:
  a       = material probability (standard)
  P_m     = physics model dependence (standard)
  0_t     = temporal offset (standard)
  0_a     = analysis framework (standard)
  θ_PL    = PL relation choice (NEW)
  θ_red   = reddening law choice (NEW)
  θ_metal = metallicity correction choice (NEW)
```

**Challenge:** How to map systematic choices to observer tensor components?

Examples:
- **PL relation choice (P1/P2/P3):**
  - Could be treated as different "physics models"
  - Higher P_m for relations with more free parameters?

- **Reddening law (MW/LMC/SMC):**
  - Different assumptions about dust properties
  - Could be captured in P_m or new θ_red component

- **Metallicity correction:**
  - Model-dependent correction
  - Affects P_m

**Problem:** These are not naturally "epistemic" differences in the same sense as CMB vs distance ladder. They're variations within a single methodology.

**Proposed solution:**
```
For each config:
  T_base = [a, P_m, 0_t, 0_a]  # Same for all distance ladder

  # Add small perturbations based on systematic choices
  δP_m += systematic_model_dependence(PL, reddening, metallicity)

  T_config = [a, P_m + δP_m, 0_t, 0_a]
```

### Estimated Weights (if implemented)

```
Configs with:
  - Simpler PL relations → higher weight
  - MW reddening (local) → higher weight
  - Standard metallicity → higher weight

Example:
  Config 1 (N,P1,MW_reddening,standard): w = 0.008 (high)
  Config 57 (M,P3,SMC_reddening,alt):    w = 0.003 (low)
```

Weighted mean would favor "simpler" systematic choices → different H₀.

---

## Why Level 1 is Currently Optimal

### Philosophical Justification

Observer tensors are meant to capture **epistemic distance** between fundamentally different measurement approaches:
- CMB (model-dependent, early universe)
- Distance ladder (direct, late universe)

Within the distance ladder, different anchors represent genuinely different **epistemic positions**:
- NGC4258: Maser kinematics (very direct)
- Milky Way: Trigonometric parallax (most direct)
- LMC: Detached eclipsing binaries (geometric but requires modeling)

These are different **ways of knowing** the distance scale.

In contrast, choosing between PL relation P1 vs P2 is not a different "way of knowing"—it's a **parametric variation within a single methodology**.

### Practical Justification

1. **Natural epistemic groups:** Anchors are distinct geometric methods
2. **Clear interpretation:** Each anchor is a different "observer"
3. **Avoids overfitting:** 3 weights vs 210 weights
4. **Computationally efficient:** Can sample posteriors per anchor
5. **Robust to systematics:** Anchor variance captures config variance

### Mathematical Justification

The anchor-level approach automatically marginalizes over systematic variations:

```
P(H₀ | anchor) = ∫ P(H₀ | config) P(config | anchor) d(config)
```

By sampling from the distribution of H₀ values within each anchor's systematic grid, we're effectively averaging over all config choices weighted by their empirical frequency.

Then we weight anchors by epistemic distance:

```
P(H₀ | all_data) = Σ w_anchor × P(H₀ | anchor)
where w_anchor ∝ 1 / (1 + |T_anchor|)
```

This is equivalent to a hierarchical model:
```
Level 1: Configs within anchors → marginalize
Level 2: Anchors → weight by epistemic distance
```

---

## If You Wanted to Implement Level 2

Here's the complete procedure:

### Step 1: Define Extended Observer Tensor

```python
def compute_config_observer_tensor(config):
    """
    Compute observer tensor for systematic configuration

    config: dict with keys ['Anc', 'PL', 'reddening', 'metallicity', ...]
    """
    # Base components (same for all distance ladder)
    a = 1 - (config['unc'] / config['H0'])  # Material probability
    P_m_base = 0.05  # Distance ladder is low model dependence
    0_t = -0.02  # Late universe
    0_a = 0.5  # Frequentist

    # Systematic choice perturbations
    δP_m = 0.0

    # PL relation complexity
    if config['PL'] == 'P1':  # Simplest
        δP_m += 0.00
    elif config['PL'] == 'P2':  # Medium
        δP_m += 0.01
    elif config['PL'] == 'P3':  # Complex
        δP_m += 0.02

    # Reddening law
    if config['reddening'] == 'MW':  # Local, well-known
        δP_m += 0.00
    elif config['reddening'] == 'LMC':  # Nearby, well-studied
        δP_m += 0.01
    elif config['reddening'] == 'SMC':  # More uncertain
        δP_m += 0.02

    # Metallicity correction
    if config['metallicity'] == 'standard':
        δP_m += 0.00
    elif config['metallicity'] == 'alternative':
        δP_m += 0.01

    P_m = P_m_base + δP_m

    return [a, P_m, 0_t, 0_a]
```

### Step 2: Compute Weights for All Configs

```python
import pandas as pd
import numpy as np

# Load systematic grid
df = pd.read_csv('data/vizier_data/J_ApJ_826_56_table3.csv')

tensors = []
weights = []

for i, row in df.iterrows():
    # Compute observer tensor for this config
    T_i = compute_config_observer_tensor(row)
    mag_i = np.linalg.norm(T_i)

    # Weight by inverse epistemic distance
    w_i = 1.0 / (1.0 + mag_i)

    tensors.append(T_i)
    weights.append(w_i)

# Normalize weights
weights = np.array(weights)
weights /= weights.sum()
```

### Step 3: Compute Weighted Mean

```python
H0_values = df['H0'].values
H0_weighted = np.sum(weights * H0_values)

print(f"Config-weighted H₀: {H0_weighted:.2f} km/s/Mpc")
```

### Step 4: Compute Weighted Uncertainty

```python
# Weighted variance
var_weighted = np.sum(weights * (H0_values - H0_weighted)**2)
unc_weighted = np.sqrt(var_weighted)

print(f"Config-weighted uncertainty: ± {unc_weighted:.2f} km/s/Mpc")
```

### Expected Result

If you implement this, you'd probably get something like:

```
H0_weighted ≈ 73.X ± 2.Y km/s/Mpc
```

Where:
- X might be slightly different from 64 (current anchor-weighted)
- Y might be slightly smaller (less anchor variance)
- Favors simpler systematic choices (lower δP_m → higher weight)

---

## Recommendation: Stick with Level 1

**For publication, I recommend:**

1. ✅ **Keep anchor-level observer tensors** (Level 1)
   - Clear epistemic interpretation
   - Avoids over-parameterization
   - Robust and reproducible

2. ✅ **Document the hierarchical structure**
   - Configs marginalized within anchors
   - Anchors weighted by epistemic distance

3. ✅ **Clarify in manuscript:**
   ```
   "Observer tensors are applied at the anchor level, with each
   anchor representing a distinct geometric calibration method.
   Systematic variations within each anchor (PL relation,
   reddening law, etc.) are marginalized via MCMC sampling from
   the empirical covariance structure."
   ```

4. ⚠️ **Optional: Sensitivity analysis**
   - Implement Level 2 as a check
   - Show that results are stable
   - Include in supplementary materials

5. ⚠️ **Future work: Level 3**
   - Process raw SNe data
   - Implement SN-level observer tensors
   - Compare with current approach
   - Publish as separate methods paper

---

**Created:** October 13, 2025
**Purpose:** Compare three levels of observer tensor application
**Recommendation:** Stick with Level 1 (anchor-level) for current publication
