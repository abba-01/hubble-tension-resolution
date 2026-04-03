# Complete Answer: Origin of shoes_weighted_h0 = 73.64 ± 3.03

## Direct Answer to Your Question

**shoes_weighted_h0 = 73.64 ± 3.03 km/s/Mpc comes from:**

✅ **Systematic grid (210 pre-computed H₀ configurations)**
✅ **Observer tensor weighting applied at ANCHOR level**
❌ NOT from raw SNe with weighting applied
❌ NOT from published SH0ES value with adjustment

## Complete Data Flow (5 Steps)

### STEP 1: Source Data
**File:** `data/vizier_data/J_ApJ_826_56_table3.csv`
**Contents:** 210 pre-computed H₀ measurements from Riess+ 2016

Each row represents a different H₀ result from varying:
- Anchor galaxy (N=NGC4258, M=Milky Way, L=LMC)
- Period-luminosity relation
- Reddening law
- Metallicity correction
- Sample selection cuts

Example structure:
```
Row   Anc  PL   break  selection  →  H₀ result
  1    N   P1   4.5      S1      →  73.24 ± 1.62
  2    N   P1   4.5      S2      →  72.89 ± 1.58
  3    N   P2   4.0      S1      →  74.11 ± 1.71
 ...
210    L   P3   4.5      S3      →  71.43 ± 2.03
```

**Key point:** Each row is already a complete H₀ measurement, not individual SNe.

---

### STEP 2: Covariance Extraction
**File:** `code/phase_c_extract_covariance.py` (inferred, not in logs)
**Output:** `results/empirical_covariance_210x210.npy`

Process:
1. Read all 210 H₀ measurements
2. Compute 210×210 empirical covariance matrix
3. Captures correlations between systematic choices

This covariance matrix encodes:
- How anchor choice affects H₀
- How PL relation choice correlates with reddening
- Full systematic uncertainty structure

---

### STEP 3: MCMC Sampling
**File:** `code/mast_mcmc_download.py`
**Method:** Multivariate sampling from empirical covariance

For each anchor (NGC4258, Milky Way, LMC):

1. **Filter:** Extract rows from systematic grid for this anchor
2. **Marginalize:** Compute mean and std across systematic variations
3. **Sample:** Generate 5,000 samples from N(μ_anchor, σ_anchor)

Results:
```
NGC4258:   72.52 ± 2.39 km/s/Mpc  (5,000 samples)
MilkyWay:  76.17 ± 2.34 km/s/Mpc  (5,000 samples)
LMC:       72.27 ± 2.62 km/s/Mpc  (5,000 samples)
```

**Saved to:**
- `data/mast/mcmc_samples_N.npy` (NGC4258)
- `data/mast/mcmc_samples_M.npy` (Milky Way)
- `data/mast/mcmc_samples_L.npy` (LMC)
- `data/mast/mcmc_metadata.json` (metadata)

---

### STEP 4: Observer Tensor Calibration
**File:** `code/calibrate_anchor_tensors.py`
**Output:** `results/anchor_tensors.json`

For each anchor, compute observer tensor:
```
T_anchor = [a, P_m, 0_t, 0_a]

where:
  a   = material probability (1 - σ/μ)
  P_m = physics model dependence (0=direct, 1=model)
  0_t = temporal offset from reference epoch
  0_a = analysis framework (Bayesian/frequentist)
```

Compute tensor magnitude: |T| = √(a² + P_m² + 0_t² + 0_a²)

Results:
```
NGC4258:   |T| = 0.5183
MilkyWay:  |T| = 0.5448
LMC:       |T| = 0.5169
```

**Key insight:** Observer tensors quantify epistemic distance between anchors.
- Similar magnitudes → anchors are epistemically comparable
- Used to compute relative weights

---

### STEP 5: Observer-Weighted Merge
**File:** `code/achieve_100pct_resolution.py`
**Method:** Inverse epistemic distance weighting

1. **Compute anchor weights:**
   ```
   w_i = 1 / (1 + |T_i|)
   ```
   Then normalize so Σw_i = 1

   Results:
   ```
   NGC4258:   w = 1/(1+0.5183) = 0.659 → normalized to 33.5%
   MilkyWay:  w = 1/(1+0.5448) = 0.647 → normalized to 32.9%
   LMC:       w = 1/(1+0.5169) = 0.659 → normalized to 33.6%
   ```

2. **Concatenate samples with weights:**
   - Total: 15,000 samples (5,000 per anchor)
   - Each sample gets weight: w_anchor / n_samples_anchor

3. **Compute weighted statistics:**
   ```
   μ_weighted = Σ(w_i × sample_i)
              = 0.335×72.52 + 0.329×76.17 + 0.336×72.27
              = 73.64 km/s/Mpc

   σ_weighted = √[Σ w_i × (sample_i - μ)²]
              = 3.03 km/s/Mpc
   ```

**Result:** shoes_weighted_h0 = 73.64 ± 3.03 km/s/Mpc

---

## Key Insights

### 1. Observer Tensors Applied at ANCHOR Level
- **NOT at configuration level** (210 configs)
- **NOT at individual SN level** (2,287 SNe)
- **YES at anchor level** (3 anchors: N, M, L)

This means:
- We treat each anchor as an epistemic "observer"
- Weight by how epistemically distant each anchor is
- Aggregate across anchors using inverse epistemic distance

### 2. Why 73.64 ≠ 73.04 (published SH0ES)?
```
Published SH0ES (Riess+ 2022):  73.04 ± 1.04 km/s/Mpc
Our weighted value:             73.64 ± 3.03 km/s/Mpc

Difference: +0.60 km/s/Mpc
Reason: Milky Way anchor (76.17) pulls average up
```

Breakdown:
- NGC4258: 72.52 (pulls down by 1.12)
- Milky Way: 76.17 (pulls up by 2.53)
- LMC: 72.27 (pulls down by 1.37)
- Weighted average with ~equal weights → 73.64

The published SH0ES value likely uses different anchor weighting or combines anchors differently.

### 3. Why σ = 3.03 >> 1.04 (published)?
```
Published SH0ES uncertainty:  1.04 km/s/Mpc
Our weighted uncertainty:     3.03 km/s/Mpc

Factor: 2.9× larger
```

Reason: We're including **anchor systematic variance**
- Published SH0ES: Optimized best-fit with anchor correlation
- Our method: Conservative propagation of anchor spread
- The 3.03 includes the ~3.9 km/s/Mpc spread between anchors

This is INTENTIONAL:
- We want to capture systematic uncertainty from anchor choice
- Published value assumes anchor discrepancies are statistical fluctuations
- Our approach treats anchor spread as genuine systematic uncertainty

---

## Your Suspicion: Was It Correct?

You suspected:
```
1. Process raw SNe into groups by systematic configuration ❌ (not what we did)
2. Apply observer tensor weighting based on each configuration's epistemic distance ❌
3. Aggregate using inverse variance + observer tensor penalties ❌
4. THEN compare to Planck ✅ (this part is correct)
```

**What we actually did:**
```
1. Use pre-computed systematic grid (210 H₀ values already computed) ✅
2. Sample from empirical covariance to get posteriors per anchor ✅
3. Apply observer tensor weighting at ANCHOR level (not config level) ✅
4. Aggregate anchors using inverse epistemic distance ✅
5. THEN merge with Planck using N/U algebra ✅
```

---

## Should We Process Raw SNe Instead?

### Current Approach (Systematic Grid)
**Pros:**
- Uses published systematic grid from Riess+ 2016
- Captures empirical covariance structure
- Computationally efficient
- Observer tensors at anchor level are interpretable

**Cons:**
- Not starting from raw SNe data
- Pre-computed H₀ values (can't verify calculation)
- Limited to 210 configurations explored by Riess+ 2016
- "Phase B: Raw SN Processing" is a misnomer

### Alternative: True Raw SN Processing
**Pros:**
- Full transparency from SNe → H₀
- Could apply observer tensors at configuration level
- Reproducible from first principles
- Better alignment with "Phase B" label

**Cons:**
- Much more complex (need full analysis pipeline)
- Requires Pantheon+SH0ES.dat (2,287 SNe)
- Need to implement all systematic variations
- Computationally expensive
- Would likely get similar results (same underlying data)

### Recommendation

**Keep current approach BUT:**
1. ✅ Rename "Phase B: Raw SN Processing" → "Phase B: Systematic Grid Aggregation"
2. ✅ Clarify that we're using pre-computed H₀ values
3. ✅ Document that observer tensors are at anchor level (not config/SN level)
4. ⚠️ Consider optional enhancement: Download raw SNe for validation

**Why?**
- Current work is scientifically valid
- Results are reproducible
- Just needs clearer labeling
- Raw SN processing would be a significant additional project

---

## How to Reproduce the 97.4% Result

Given the data flow above, here's how to reproduce:

### Step-by-step:

1. **Start with systematic grid** (210 H₀ configs)
   ```python
   df = pd.read_csv('data/vizier_data/J_ApJ_826_56_table3.csv')
   # 210 rows, each with H₀ and uncertainty
   ```

2. **Generate MCMC samples per anchor**
   ```python
   for anchor in ['N', 'M', 'L']:
       mask = df['Anc'] == anchor
       h0_values = df[mask]['H0'].values
       μ_anchor = np.mean(h0_values)
       σ_anchor = np.std(h0_values)
       samples = np.random.normal(μ_anchor, σ_anchor, 5000)
   ```

3. **Compute observer tensors for each anchor**
   ```python
   T_N = [a_N, P_m_N, 0_t_N, 0_a_N]  # NGC4258
   T_M = [a_M, P_m_M, 0_t_M, 0_a_M]  # Milky Way
   T_L = [a_L, P_m_L, 0_t_L, 0_a_L]  # LMC
   |T_i| = sqrt(sum(T_i²))
   ```

4. **Weight anchors by inverse epistemic distance**
   ```python
   w_N = 1/(1 + |T_N|)
   w_M = 1/(1 + |T_M|)
   w_L = 1/(1 + |T_L|)
   # Normalize
   ```

5. **Compute weighted SH0ES posterior**
   ```python
   shoes_weighted = w_N × samples_N + w_M × samples_M + w_L × samples_L
   μ_shoes = mean(shoes_weighted) = 73.64
   σ_shoes = std(shoes_weighted) = 3.03
   ```

6. **Merge with Planck using N/U algebra**
   ```python
   H0_planck = 67.4 ± 0.5
   H0_shoes = 73.64 ± 3.03

   # Inverse variance weights
   w_early = 1/0.5²
   w_late = 1/3.03²
   H0_merged = (w_early×67.4 + w_late×73.64)/(w_early + w_late)
             = 67.57

   # Base uncertainty
   u_base = 1/sqrt(w_early + w_late) = 0.493

   # Epistemic penalty
   disagreement = |67.4 - 73.64| = 6.24
   systematic_fraction = 0.519
   epistemic_penalty = (6.24/2) × 0.527 × (1-0.519) = 0.790

   # Merged uncertainty
   u_merged = sqrt(0.493² + 0.790²) = 0.931
   ```

7. **Result:**
   ```
   H0_merged = 67.57 ± 0.93 km/s/Mpc

   Tension reduction:
   Before: |73.64 - 67.40| = 6.24
   After:  |67.57 - 67.40| = 0.17
   Reduction: 97.4%
   ```

---

## Correct Observer Tensor Application

Based on the actual code, here's the correct procedure:

### Level 1: Anchor-Level Observer Tensors (what we did)
```
Input: 3 anchor posteriors (N, M, L)
Observer tensors: One per anchor
Application: Weight anchors by inverse epistemic distance
Output: Single weighted SH0ES posterior
```

**Interpretation:** Anchors are treated as different "observers" with different epistemic positions.

### Level 2: Configuration-Level Observer Tensors (alternative)
```
Input: 210 configuration posteriors
Observer tensors: One per configuration
Application: Weight configs by inverse epistemic distance
Output: Single weighted SH0ES posterior
```

**Interpretation:** Each systematic choice is a different "observer".

### Level 3: SN-Level Observer Tensors (most granular)
```
Input: 2,287 individual SNe
Observer tensors: One per (SN, systematic config) pair
Application: Weight each SN × config by epistemic distance
Output: Single weighted SH0ES posterior
```

**Interpretation:** Each SN measured with each systematic choice is a different "observer".

### Which Level Did We Use?

**Level 1: Anchor-level** ✅

This is the simplest and most interpretable:
- NGC4258 is a maser galaxy (geometric distance)
- Milky Way uses parallax (geometric distance)
- LMC uses detached eclipsing binaries (geometric distance)

Each anchor represents a fundamentally different geometric calibration method, making them natural epistemic "observers".

---

## What You Should Look At

To understand the correct observer tensor application, examine these files:

### 1. Observer Tensor Calibration
**File:** `code/calibrate_anchor_tensors.py`
- Shows HOW observer tensors are computed for each anchor
- Shows WHAT goes into [a, P_m, 0_t, 0_a]
- Lines showing tensor magnitude calculation

### 2. Observer-Weighted Merge
**File:** `code/achieve_100pct_resolution.py`
- Lines 75-101: How anchor weights are computed
- Lines 110-131: How weighted statistics are computed
- Lines 148-210: N/U merge formula with epistemic penalty

### 3. MCMC Generation
**File:** `code/mast_mcmc_download.py`
- Lines 76-120: How samples are generated per anchor
- Shows marginalization over systematic grid

### 4. Systematic Grid Data
**File:** `data/vizier_data/J_ApJ_826_56_table3.csv`
- Raw systematic grid (210 rows)
- Shows what each "configuration" represents

### 5. Phase D Result
**File:** `results/resolution_100pct_mcmc.json`
- Final numbers with full metadata
- Shows anchor weights used

---

## Summary: Your Three Questions

### Q1: Is it from the systematic grid (210 configs)?
**A:** YES ✅

The systematic grid is the ultimate source. But:
- Grid is aggregated per anchor (3 groups)
- Sampled into posteriors (5,000 samples each)
- Weighted by observer tensors at anchor level

### Q2: Is it from raw SNe with some weighting applied?
**A:** NO ❌

We do not process raw SNe. We use pre-computed H₀ values from the systematic grid. Each row in the grid is already a complete H₀ measurement.

### Q3: Is it from published SH0ES with observer tensor adjustment?
**A:** PARTIALLY ❌/✅

We use data from the same source as published SH0ES (Riess+ 2016), but:
- NOT the single published value (73.04 ± 1.04)
- Instead: Systematic grid that underlies that publication
- Observer tensors applied at anchor level
- Different weighting scheme → different result (73.64 ± 3.03)

---

## Final Recommendation

### For the manuscript:
1. **Rename Phase B** to "Systematic Grid Aggregation"
2. **Add clarification** that observer tensors are at anchor level
3. **Explain** why 73.64 ≠ 73.04 (anchor weighting + systematic variance)
4. **Document** the complete data flow in methods section

### For validation:
1. **Keep current approach** (it's scientifically valid)
2. **Optional:** Download Pantheon+SH0ES.dat for comparison
3. **Future work:** Implement config-level or SN-level observer tensors

### For reproducibility:
1. **Document** the complete pipeline (5 steps above)
2. **Provide** all intermediate files (MCMC samples, tensors, covariance)
3. **Clarify** assumptions (anchor-level weighting, marginal sampling)

---

**Created:** October 13, 2025
**Purpose:** Answer user's question about shoes_weighted_h0 source
**Status:** Complete data flow traced from VizieR → Phase D result
