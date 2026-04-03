# APPENDIX B: STABILITY VALIDATION PRE-FLIGHT CHECKLIST

**Purpose:** Comprehensive verification before executing the full Appendix B pipeline  
**Version:** 1.0.0  
**Date:** 2025-10-12

---

## ✅ SECTION 1: ENVIRONMENT INTEGRITY

### 1.1 Python Version
```bash
python3 --version
```
- [ ] **Expected:** Python 3.8 or higher
- [ ] **Actual version:** _______________

### 1.2 Required Packages
```bash
pip show numpy pandas matplotlib
```
- [ ] **numpy** installed (version: ________)
- [ ] **pandas** installed (version: ________)
- [ ] **matplotlib** installed (version: ________)

**If missing, install:**
```bash
pip install numpy pandas matplotlib
```

### 1.3 Script Files Present
Check all three scripts are in the working directory:

```bash
ls -lh *.py
```

- [ ] **sensitivity_analysis.py** (expected ~5-8 KB)
- [ ] **monte_carlo_validation.py** (expected ~6-10 KB)
- [ ] **appendix_b_stability_validation.py** (expected ~8-12 KB)

### 1.4 Write Permissions
```bash
touch ._test_write && rm ._test_write && echo "✓ Write OK"
```
- [ ] **Working directory is writable**

---

## ✅ SECTION 2: DEPENDENCY & INTERFACE SANITY

### 2.1 Expected Output Files

**After `sensitivity_analysis.py` runs:**
- [ ] Creates `deltaT_sensitivity_results.csv`
- [ ] CSV contains columns: `ΔT`, `u_expand`, `u_merged`, `gap`, `resolution_%`, `concordance`
- [ ] At least one row has `concordance` = '✓'

**After `monte_carlo_validation.py` runs:**
- [ ] Creates `monte_carlo_results.json`
- [ ] JSON contains keys: `n_samples`, `both_containment`, `random_seed`
- [ ] Creates `monte_carlo_validation.png`

### 2.2 Error Handling Verification

**Check `appendix_b_stability_validation.py` contains:**

```python
# After loading sensitivity_df
concordant = sensitivity_df[sensitivity_df['concordance'] == '✓']
if concordant.empty:
    sys.exit("No ΔT value achieved full concordance.")
```
- [ ] **Concordance check present**

```python
# After loading mc_results
if 'both_containment' not in mc_results or 'n_samples' not in mc_results:
    sys.exit("Monte Carlo results JSON missing expected keys.")
```
- [ ] **JSON key validation present**

---

## ✅ SECTION 3: EXECUTION MODE

### 3.1 Debug vs Production Mode

**For debugging (live output):**
```python
result = subprocess.run([sys.executable, "script.py"])
```

**For archival (silent, logged):**
```python
result = subprocess.run([sys.executable, "script.py"], 
                       capture_output=True, text=True, check=True)
```

- [ ] **Current mode:** ☐ Debug  ☐ Production
- [ ] **Mode is appropriate for current phase**

---

## ✅ SECTION 4: REPRODUCIBILITY METADATA

### 4.1 Random Seed Fixed
```python
random_seed = 20251012  # In monte_carlo_validation.py
np.random.seed(random_seed)
```
- [ ] **Seed is hardcoded (20251012)**

### 4.2 Software Versions Recorded
Check `appendix_b_stability_validation.py` includes:
```python
summary["random_seed"] = 20251012
summary["software_versions"] = {
    "python": sys.version,
    "numpy": np.__version__,
    "pandas": pd.__version__
}
```
- [ ] **Metadata capture present**

---

## ✅ SECTION 5: DRY RUN (OPTIONAL BUT RECOMMENDED)

### 5.1 Test Sensitivity Analysis Alone
```bash
python sensitivity_analysis.py
```
**Expected output:**
- Console displays full sensitivity grid
- Message: "✓ Results saved to: deltaT_sensitivity_results.csv"
- File created with ~13 rows (ΔT from 1.0 to 1.6)

- [ ] **Script runs without errors**
- [ ] **Output file created**
- [ ] **At least one '✓' in concordance column**

### 5.2 Test Monte Carlo Alone
```bash
python monte_carlo_validation.py
```
**Expected output:**
- Progress indicators: "X / 10,000 samples processed..."
- Message: "✓ Visualization saved to: monte_carlo_validation.png"
- Message: "✓ Results saved to: monte_carlo_results.json"

- [ ] **Script runs without errors**
- [ ] **JSON created with correct keys**
- [ ] **PNG visualization generated**

### 5.3 Validate JSON Structure
```bash
python -c "import json; data=json.load(open('monte_carlo_results.json')); print('✓' if 'both_containment' in data else '✗')"
```
- [ ] **Returns:** ✓

---

## ✅ SECTION 6: FULL PIPELINE EXECUTION

### 6.1 Clean Start (Optional)
```bash
rm -f deltaT_sensitivity_results.csv monte_carlo_results.json \
      monte_carlo_validation.png appendix_b_stability.md \
      stability_summary.json
```
- [ ] **Old files removed (if doing fresh run)**

### 6.2 Execute Combined Script
```bash
python appendix_b_stability_validation.py
```

**Monitor for:**
- Section 0: "✓ All environment checks passed"
- Section 1: "✓ Sensitivity analysis complete"
- Section 2: "✓ Monte Carlo validation complete"
- Section 3: Numerical targets table
- Section 4: "✓ Appendix B drafted"
- Section 5: "✓ All output files verified"

- [ ] **All sections complete without errors**

---

## ✅ SECTION 7: OUTPUT VERIFICATION

### 7.1 Required Files Present
```bash
ls -lh *.csv *.json *.png *.md
```

Check all five files exist and are non-empty:

- [ ] `deltaT_sensitivity_results.csv` (>1 KB)
- [ ] `monte_carlo_results.json` (>500 bytes)
- [ ] `monte_carlo_validation.png` (>50 KB)
- [ ] `appendix_b_stability.md` (>3 KB)
- [ ] `stability_summary.json` (>500 bytes)

### 7.2 Verify Summary Verdict
```bash
grep "overall_verdict" stability_summary.json
```

- [ ] **Returns:** "BULLETPROOF" or "PUBLICATION_READY" or "REQUIRES_REFINEMENT"
- [ ] **Verdict matches expectations:** _________________

### 7.3 Check Numerical Targets

**Expected from summary JSON:**
```json
{
  "numerical_targets": {
    "critical_deltaT_achieved": ~1.10,
    "stability_margin_achieved": >20.0,
    "mc_frequency_achieved": >=0.95
  },
  "all_targets_met": true
}
```

Extract and verify:
```bash
python -c "import json; s=json.load(open('stability_summary.json')); \
           print(f\"Critical ΔT: {s['sensitivity_analysis']['critical_deltaT']:.2f}\"); \
           print(f\"Margin: {s['sensitivity_analysis']['stability_margin_percent']:.1f}%\"); \
           print(f\"MC freq: {s['monte_carlo_validation']['concordance_frequency']:.4f}\")"
```

- [ ] **Critical ΔT:** _______ (target: 1.08-1.12)
- [ ] **Stability margin:** _______ % (target: >20%)
- [ ] **MC frequency:** _______ (target: ≥0.95)

---

## ✅ SECTION 8: ARCHIVAL STEP

### 8.1 Generate Checksums
```bash
sha256sum appendix_b_stability_validation.py \
          sensitivity_analysis.py \
          monte_carlo_validation.py \
          stability_summary.json \
          > run_hash.txt
```
- [ ] **Checksum file created**

### 8.2 Document Environment
```bash
echo "Python: $(python3 --version)" >> run_hash.txt
echo "NumPy: $(python3 -c 'import numpy; print(numpy.__version__)')" >> run_hash.txt
echo "Pandas: $(python3 -c 'import pandas; print(pandas.__version__)')" >> run_hash.txt
echo "Date: $(date -Iseconds)" >> run_hash.txt
```
- [ ] **Environment metadata appended**

### 8.3 Verify run_hash.txt
```bash
cat run_hash.txt
```
- [ ] **Contains SHA-256 hashes**
- [ ] **Contains software versions**
- [ ] **Contains execution timestamp**

---

## ✅ SECTION 9: PUBLICATION READINESS

### 9.1 Review Appendix B Draft
```bash
head -n 30 appendix_b_stability.md
```

Check for:
- [ ] Execution timestamp present
- [ ] Random seed documented (20251012)
- [ ] Software versions listed
- [ ] Critical ΔT reported
- [ ] MC frequency reported
- [ ] Overall verdict stated
- [ ] Reproducibility instructions included

### 9.2 Check Visualization
```bash
file monte_carlo_validation.png
```
- [ ] **Returns:** PNG image data, ~800x400 or similar
- [ ] **Open image to verify:** Histogram + bar chart visible

### 9.3 Final Artifact List for Supplementary Materials

**To include in publication package:**

1. ☐ `appendix_b_stability.md` (main text)
2. ☐ `deltaT_sensitivity_results.csv` (Table B.1)
3. ☐ `monte_carlo_validation.png` (Figure B.1)
4. ☐ `stability_summary.json` (metadata)
5. ☐ `run_hash.txt` (checksums + environment)
6. ☐ All three Python scripts (code repository)

---

## ✅ SECTION 10: SUCCESS CRITERIA

### 10.1 Minimum Requirements (Publication-Ready)
- [ ] Critical ΔT in range [1.08, 1.12]
- [ ] Stability margin ≥ 10%
- [ ] MC concordance frequency ≥ 0.90
- [ ] All output files generated without errors
- [ ] Checksums computed and documented

### 10.2 Target Requirements (Statistically Bulletproof)
- [ ] Critical ΔT in range [1.08, 1.12]
- [ ] Stability margin ≥ 20%
- [ ] MC concordance frequency ≥ 0.95
- [ ] All tests pass with ✓ status
- [ ] Overall verdict: "BULLETPROOF"

---

## SIGN-OFF

**Pre-flight check completed by:** _________________  
**Date:** _________________  
**All critical items verified:** ☐ YES  ☐ NO  

**If NO, specify issues:**
_______________________________________________
_______________________________________________

**Approval to proceed:** ☐ YES  ☐ NO  

---

## TROUBLESHOOTING GUIDE

### Common Issues

**Issue 1:** "No ΔT value achieved full concordance"
- **Cause:** Empirical ΔT may be too low
- **Fix:** Check tensor assignments in Section 5.2 of SSoT

**Issue 2:** "Monte Carlo results JSON missing expected keys"
- **Cause:** Script modified or incomplete execution
- **Fix:** Re-run `monte_carlo_validation.py` alone first

**Issue 3:** MC frequency < 0.95
- **Cause:** Conservative bounds may be too tight
- **Fix:** Review N/U algebra uncertainty propagation

**Issue 4:** Stability margin < 20%
- **Cause:** Critical ΔT too close to empirical value
- **Fix:** Refine tensor calibration with MCMC data

---

**END OF CHECKLIST**

*Verify all ✓ boxes are checked before proceeding to full execution*
