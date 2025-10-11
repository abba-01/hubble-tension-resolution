# Claude Code Task: Bootstrap Validation Execution

## Objective
Set up and execute the bootstrap validation for the Hubble tension observer-tensor framework in the `hubble_montecarlo_package2_20251011` repository.

## Prerequisites
- Repository location: `hubble_montecarlo_package2_20251011`
- Python 3.10+ installed
- Git configured for private repository

## Task Sequence

### Phase 1: Repository Setup (5 minutes)
```bash
# Navigate to repository
cd hubble_montecarlo_package2_20251011

# Run setup script
python bootstrap_setup_exec.py

# Install dependencies
pip install -r requirements.txt
```

**Expected Output:**
- Directory structure created
- All files generated
- Checksum displayed for CORRECTED_RESULTS_32BIT.json

**Validation Checkpoint:**
- [ ] All directories exist (code/, data/, validation_results/, docs/)
- [ ] CORRECTED_RESULTS_32BIT.json created (~8 KB)
- [ ] bootstrap_validation.py created (~12 KB)
- [ ] requirements.txt exists

### Phase 2: Execute Bootstrap Validation (5-10 minutes)
```bash
cd code
python bootstrap_validation.py
```

**Expected Output:**
```
Loaded 6 probes from CORRECTED_RESULTS_32BIT.json
  Early group: 2 probes
  Late group: 4 probes
Checksum: [hash]...

Generating 10,000 bootstrap samples (seed=20251011)...
  ✓ Planck18: 10,000 samples
  ✓ DES-IDL: 10,000 samples
  ✓ SH0ES: 10,000 samples
  ✓ TRGB: 10,000 samples
  ✓ TDCOSMO: 10,000 samples
  ✓ Megamaser: 10,000 samples

Computing tensor-weighted merges for 10,000 iterations...
  Progress: 2,000/10,000 (20.0%)
  Progress: 4,000/10,000 (40.0%)
  Progress: 6,000/10,000 (60.0%)
  Progress: 8,000/10,000 (80.0%)
  Progress: 10,000/10,000 (100.0%)

✓ Saved: ../validation_results/bootstrap_samples.csv
✓ Saved: ../validation_results/validation_summary.json
✓ Saved: ../validation_results/reproducibility.yaml

VALIDATION SUMMARY
Gap: 0.XX ± 0.XX km/s/Mpc
95% CI: [0.XX, 0.XX]
Target gap: 0.48 km/s/Mpc
Reduction achieved: XX.X%

Runtime: XXX.X seconds
```

**Validation Checkpoint:**
- [ ] bootstrap_samples.csv created (~2 MB, 10,000 rows)
- [ ] validation_summary.json created
- [ ] reproducibility.yaml created
- [ ] No Python errors
- [ ] Runtime < 600 seconds

### Phase 3: Results Analysis
```bash
# View summary
cat ../validation_results/validation_summary.json

# Check file sizes
ls -lh ../validation_results/

# Verify CSV structure
head -n 5 ../validation_results/bootstrap_samples.csv
```

**Critical Metrics to Report:**
1. **Gap Statistics:**
   - Mean: [target: 0.48 km/s/Mpc]
   - Std: [expected: ~0.10-0.15]
   - 95% CI: [should span 0.24-0.72]

2. **Validation Flags:**
   - mean_bias_ok: [should be true]
   - no_outliers: [should be true]
   - reduction_from_original: [target: ~0.91]

3. **Reproducibility:**
   - Runtime: [expected: 200-400 seconds]
   - Checksums: [document for verification]

### Phase 4: Git Commit & Push
```bash
cd ..

# Stage files
git add .
git status

# Commit
git commit -m "Bootstrap validation Phase A: Statistical reproducibility

- Created canonical data file (CORRECTED_RESULTS_32BIT.json)
- Implemented bootstrap validation pipeline
- Generated 10,000 sample validation
- Results: [XX.X]% tension reduction validated
- Runtime: [XXX]s, seed: 20251011"

# Push to private repository
git push origin main
```

**Validation Checkpoint:**
- [ ] All new files staged
- [ ] Commit message descriptive
- [ ] Push successful
- [ ] Private repository updated

## Detailed Reporting Requirements

### Report Section 1: Setup Verification
```
Directory Structure:
- [✓/✗] code/ created
- [✓/✗] validation_results/ created
- [✓/✗] CORRECTED_RESULTS_32BIT.json (size: ___ bytes)
- [✓/✗] bootstrap_validation.py (size: ___ bytes)

Data Integrity:
- File checksum: [first 12 chars]
- Probes loaded: [count]
- Early group: [names]
- Late group: [names]
```

### Report Section 2: Execution Metrics
```
Bootstrap Sampling:
- Iterations: 10,000
- Seed: 20251011
- Sampling method: Normal(n, u)
- Completion: [timestamp]

Computational Performance:
- Total runtime: ___ seconds
- Samples/second: ___ 
- Memory peak: ___ MB
- CPU usage: ___% average

Progress Checkpoints:
- 20% complete: [timestamp]
- 40% complete: [timestamp]
- 60% complete: [timestamp]
- 80% complete: [timestamp]
- 100% complete: [timestamp]
```

### Report Section 3: Statistical Results
```
Primary Metrics:
┌─────────────┬────────┬──────┬─────────────────┐
│ Parameter   │ Mean   │ Std  │ 95% CI          │
├─────────────┼────────┼──────┼─────────────────┤
│ H0_early    │ XX.XX  │ X.XX │ [XX.XX, XX.XX]  │
│ H0_late     │ XX.XX  │ X.XX │ [XX.XX, XX.XX]  │
│ delta_T     │ X.XXX  │ X.XX │ [X.XXX, X.XXX]  │
│ H0_merged   │ XX.XX  │ X.XX │ [XX.XX, XX.XX]  │
│ gap         │ X.XX   │ X.XX │ [X.XX, X.XX]    │
└─────────────┴────────┴──────┴─────────────────┘

Validation Checks:
- Mean bias < 0.05: [✓/✗] (actual: ___)
- Target gap (0.48): [✓/✗] (actual: ___)
- No outliers > 5σ: [✓/✗]
- Reduction from 5.40: ___% [✓/✗] (target: ~91%)

Published Value Comparison:
- Early H0: Published=67.32, Bootstrap=XX.XX [✓/✗ within 95% CI]
- Late H0: Published=72.72, Bootstrap=XX.XX [✓/✗ within 95% CI]
- Gap: Published=0.48, Bootstrap=X.XX [✓/✗ within tolerance]
```

### Report Section 4: Output Files
```
Generated Files:
1. bootstrap_samples.csv
   - Size: ___ MB
   - Rows: 10,000
   - Columns: 5 (H0_early, H0_late, delta_T, H0_merged, gap)
   - First row: [paste first data row]
   - Checksum: [12 chars]

2. validation_summary.json
   - Size: ___ bytes
   - Structure: [paste formatted JSON]
   - Checksum: [12 chars]

3. reproducibility.yaml
   - Size: ___ bytes
   - Seed: 20251011
   - Python: ___ 
   - NumPy: ___
   - Runtime: ___ seconds
```

### Report Section 5: Issues & Resolutions
```
Encountered Issues:
1. [Issue description]
   Resolution: [action taken]
   Status: [resolved/unresolved]

2. [...]

Warnings:
- [any Python warnings]
- [any numerical warnings (NaN, inf, etc.)]

Error Messages:
- [any errors encountered]
- [stack traces if applicable]
```

### Report Section 6: Next Steps Assessment
```
Phase A Completion:
- [✓/✗] Statistical reproducibility validated
- [✓/✗] 91% reduction claim confirmed
- [✓/✗] Output files meet specifications
- [✓/✗] Repository updated

Ready for Phase B (MCMC Calibration)?
- [yes/no] [reasoning]

Recommended Actions:
1. [action item]
2. [action item]
3. [action item]
```

## Success Criteria

**PASS if ALL true:**
- ✓ All files created without errors
- ✓ Bootstrap completed 10,000 iterations
- ✓ Gap mean within [0.43, 0.53] km/s/Mpc
- ✓ Reduction ≥ 89% (allowing 2% margin)
- ✓ No validation flags failed
- ✓ Repository pushed successfully

**INVESTIGATE if ANY true:**
- Gap mean outside [0.40, 0.56]
- Reduction < 89% or > 95%
- Runtime > 600 seconds
- Outliers detected
- Checksum mismatch on re-run

## Contact & Escalation

If execution fails or results are unexpected:
1. Capture full error output
2. Document exact failure point
3. Preserve all generated files
4. Report complete state to human operator

## Execution Timeline

- Setup: 5 minutes
- Validation: 5-10 minutes
- Analysis: 2 minutes
- Git commit: 2 minutes
- **Total: ~15-20 minutes**

---

**Task Priority:** HIGH
**Execution Mode:** Sequential (each phase gates next)
**Failure Handling:** Stop and report on first error
**Output Format:** Detailed markdown report with all metrics
