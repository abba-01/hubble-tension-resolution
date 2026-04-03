# Quick Start Guide

## 🚀 Get Results in 3 Minutes

### Step 1: Install Dependencies (30 seconds)
```bash
pip install numpy pandas scipy matplotlib
```

### Step 2: Run the Analysis (2 minutes)
```bash
# Navigate to package directory
cd hubble_tensor_calibration_package

# Run the complete analysis
python3 core/multiresolution_uha_tensor_calibration.py
```

### Step 3: View Results (30 seconds)
```bash
# Generate comparison plots
python3 scripts/generate_comparison.py

# View validation report
python3 core/validate_multiresolution.py
```

## ✅ Expected Output

You should see:
```
================================================================================
================================================================================

>>> Processing at 8-bit resolution...
  Δ_T = 0.008213 (was stuck at 0.6255!)
  
>>> Processing at 16-bit resolution...
  Δ_T = 0.008109 (was stuck at 0.6255!)
  
>>> Processing at 21-bit resolution...
  Δ_T = 0.008108 (was stuck at 0.6255!)
  
✓ CONVERGED at 21-bit resolution!
Final H₀ = 68.52 ± 0.45 km/s/Mpc
```

## 🔑 Key Points

1. **Δ_T must change with resolution** - If it stays at 0.6255, you're using the old broken version
2. **Convergence at 21-bit** - This is expected and correct
3. **H₀ ~ 68.5** - Should be between Planck (67.4) and SH0ES (73.0)

## 📊 Check Your Results

Results are saved in:
- `results/multiresolution_results.json` - Numerical results
- `visualizations/convergence_plot.png` - Visual proof of convergence
- `visualizations/before_after_comparison.png` - Shows the fix

## ⚠️ Troubleshooting

If Δ_T stays at 0.6255:
- You're using single resolution (broken version)
- Check that `resolution_levels = [8, 16, 21, 32]` in the code

If no convergence:
- Increase iterations or add 32-bit resolution
- Check that learning rate alpha = 0.15

## 🎯 Success!

When you see "CONVERGED at 21-bit resolution" with Δ_T ~ 0.008, the methodology is working correctly!
