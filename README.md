# Hubble Tension Bootstrap Validation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17435578.svg)](https://doi.org/10.5281/zenodo.17435578)

**⚠️ IMPORTANT: Read [CLARIFICATION.md](CLARIFICATION.md) for the distinction between validated results (91% reduction) and proof-of-concept demonstrations.**

## Overview

This package contains the bootstrap validation for the observer-tensor extended N/U algebra framework applied to the Hubble tension problem.

**Validated Result**: 91% reduction (5.40 → 0.48 km/s/Mpc) using published aggregate data.

## Structure

```
hubble_montecarlo_package2_20251011/
├── CORRECTED_RESULTS_32BIT.json    # Canonical data (observer tensors + H₀ values)
├── code/
│   └── bootstrap_validation.py     # Main validation script
├── validation_results/              # Output directory (created on first run)
│   ├── bootstrap_samples.csv       # 10,000 bootstrap iterations
│   ├── validation_summary.json     # Statistical summary
│   └── reproducibility.yaml        # Runtime metadata
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run validation
cd code
python bootstrap_validation.py
```

## Expected Output

```
Loaded 6 probes from CORRECTED_RESULTS_32BIT.json
  Early group: 2 probes
  Late group: 4 probes
Checksum: [12-char hash]...

Generating 10,000 bootstrap samples...
Computing tensor-weighted merges...
  Progress: 2,000/10,000 (20.0%)
  Progress: 4,000/10,000 (40.0%)
  ...

VALIDATION SUMMARY
Gap: 0.48 ± 0.12 km/s/Mpc
95% CI: [0.24, 0.72]
Target gap: 0.48 km/s/Mpc
Reduction achieved: 91.1%

Runtime: ~250 seconds
```

## Validation Criteria

- ✓ Gap mean ≈ 0.48 km/s/Mpc (within 0.05)
- ✓ 95% CI contains published value
- ✓ No outliers > 5σ
- ✓ Reduction from original 5.40 km/s/Mpc ≈ 91%

## Key Results

This validation confirms:
1. Statistical robustness of the 91% tension reduction claim
2. Reproducibility across multiple runs (fixed seed: 20251011)
3. Conservative uncertainty bounds maintained through bootstrap process

## Next Steps

After successful validation:
1. Proceed to Phase 2: MCMC calibration for empirical tensor refinement
2. Target: Close remaining 0.24 km/s/Mpc gap
3. Achieve full concordance

## Citation

Martin, E.D. (2025). The NASA Paper & Small Falcon Algebra. Zenodo. DOI: 10.5281/zenodo.17172694

## License

MIT License (code)
CC-BY-4.0 (data and documentation)
