#!/bin/bash
# MASTER EXECUTION: 100% Hubble Tension Resolution
# Uses MAST MCMC chains + N/U Algebra + Observer Tensors

set -e

cd /run/media/root/OP01/got/hubble

echo "================================================================================"
echo "100% HUBBLE TENSION RESOLUTION PIPELINE"
echo "================================================================================"
echo "Objective: Download MCMC chains and achieve >95% tension reduction"
echo "Method: N/U Algebra + Observer Domain Tensors + Full Posterior"
echo ""
echo "Time: $(date)"
echo "================================================================================"
echo ""

# Step 1: Download MCMC chains from MAST
echo "[STEP 1/2] Downloading SH0ES MCMC chains from MAST..."
python3 <<'PYEOF'
import sys
sys.path.insert(0, 'code')
exec(open('code/mast_mcmc_download.py').read())
PYEOF

echo ""
echo "✅ STEP 1 COMPLETE"
echo ""

# Step 2: Apply framework for 100% resolution
echo "[STEP 2/2] Applying N/U + Tensors framework to full posterior..."
python3 <<'PYEOF'
import sys
sys.path.insert(0, 'code')
exec(open('code/achieve_100pct_resolution.py').read())
PYEOF

echo ""
echo "✅ STEP 2 COMPLETE"
echo ""

# Display results
echo "================================================================================"
echo "PIPELINE COMPLETE - CHECKING RESULTS"
echo "================================================================================"

if [ -f "results/resolution_100pct_mcmc.json" ]; then
    echo ""
    echo "📊 FINAL RESULTS:"
    python3 -c "
import json
with open('results/resolution_100pct_mcmc.json') as f:
    r = json.load(f)
    print(f\"   Baseline gap: {r['results']['gap_baseline']:.2f} km/s/Mpc\")
    print(f\"   After merge: {r['results']['gap_after_merge']:.2f} km/s/Mpc\")
    print(f\"   REDUCTION: {r['results']['reduction_percent']:.1f}%\")
    print(f\"\")
    print(f\"   Merged H0: {r['results']['merged_h0']:.2f} ± {r['results']['merged_uncertainty']:.2f} km/s/Mpc\")
    if r['results']['reduction_percent'] >= 95:
        print(f\"\")
        print(f\"   🎉 OBJECTIVE ACHIEVED: >95% REDUCTION\")
    elif r['results']['reduction_percent'] >= 85:
        print(f\"\")
        print(f\"   ✅ SUBSTANTIAL REDUCTION ACHIEVED\")
"
    echo ""
else
    echo "❌ Results file not found"
    exit 1
fi

echo "================================================================================"
echo "All results saved in:"
echo "  results/resolution_100pct_mcmc.json"
echo "  data/mast/*.npy"
echo "================================================================================"
