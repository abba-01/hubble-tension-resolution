# APPENDIX B: STABILITY VALIDATION

**Execution timestamp:** 2025-10-12T21:32:24.714108  
**Random seed:** 20251012 (for exact reproducibility)  
**Software versions:** Python 3.12.9, NumPy 1.26.4, Pandas 2.2.1

## B.1 Sensitivity Analysis

**Critical ΔT threshold:** 1.10

The minimum epistemic distance required for 100% concordance is ΔT ≥ 1.10.
The empirical value (ΔT = 1.3477) provides a 22.5% stability 
margin, indicating robust parametric robustness.

**Interpretation:** The framework is robust to ±11.3% variations in tensor components.

**Full sensitivity grid:** See Table B.1 (deltaT_sensitivity_results.csv)

## B.2 Monte Carlo Validation

**Concordance frequency:** 0.7843 (78.43%)  
**Samples:** 10,000  
**Random seed:** 20251012

Over 10,000 stochastic samples, the merged interval contained both 
early and late universe measurements in 78.43% of cases, demonstrating 
poor stochastic alignment with the deterministic framework.

**Interpretation:** The bounds may require refinement for optimal coverage.

**Detailed results:** See monte_carlo_results.json and Figure B.1 (monte_carlo_validation.png)

## B.3 Integrated Assessment

The framework demonstrates:

1. **Parametric stability:** 22.5% margin above critical threshold
   - Status: ✓ ROBUST
   - The 22.5% margin provides strong stability against tensor calibration uncertainty

2. **Stochastic validation:** 78.43% containment frequency
   - Status: ✗ POOR
   - Significant mismatch between deterministic and stochastic

**Overall verdict:** ⚠ REQUIRES REFINEMENT - One or both metrics below target thresholds.

## B.4 Reproducibility Guarantee

All analyses are fully deterministic and reproducible:

- **Fixed random seed:** 20251012
- **Software versions:** Documented above
- **Input data:** Six published H₀ measurements (unchanged)
- **Algorithms:** Open-source implementations (see code repository)

To reproduce exactly:

```bash
python sensitivity_analysis.py
python monte_carlo_validation.py
python appendix_b_stability_validation.py
```

Expected checksums:
- `deltaT_sensitivity_results.csv`: [to be computed]
- `monte_carlo_results.json`: [to be computed]
- `stability_summary.json`: [to be computed]

## B.5 Limitations and Future Work

Current limitations:

1. Observer tensor assignments are based on methodology inference, not direct MCMC extraction
2. Monte Carlo assumes Gaussian distributions (conservative but may overestimate tails)
3. Only six H₀ probes included (more probes would reduce statistical uncertainty)

Future refinements:

1. Extract empirical tensors from full Planck and SH0ES MCMC chains
2. Test alternative distributions (Student's t, uniform) in Monte Carlo
3. Include additional probes (JWST Cepheids, Roman weak lensing, 21cm cosmology)
