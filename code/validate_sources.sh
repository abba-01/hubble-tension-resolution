#!/bin/bash
# Pre-Flight Validation Script
# Purpose: Verify all source files exist before consolidation
# Date: October 14, 2025

set -e
umask 022

SOURCE_REPO="/run/media/root/OP01/got/hubble"
SOURCE_MAST="/run/media/root/OP01/got/hubble_data/mast_anchors"
VALIDATION_LOG="validation_$(date +%Y%m%d_%H%M%S).log"

echo "=======================================================================" | tee "$VALIDATION_LOG"
echo "Pre-Flight Validation" | tee -a "$VALIDATION_LOG"
echo "=======================================================================" | tee -a "$VALIDATION_LOG"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$VALIDATION_LOG"
echo "" | tee -a "$VALIDATION_LOG"

# Counters
FILES_FOUND=0
FILES_MISSING=0

# Function to check file existence
check_file() {
    local filepath="$1"
    if [ -f "$filepath" ]; then
        echo "  ✓ Found: $filepath" | tee -a "$VALIDATION_LOG"
        ((FILES_FOUND++))
    else
        echo "  ✗ MISSING: $filepath" | tee -a "$VALIDATION_LOG"
        ((FILES_MISSING++))
    fi
}

# Check Phase Scripts
echo "Checking Phase Scripts..." | tee -a "$VALIDATION_LOG"
check_file "$SOURCE_REPO/scripts/run_phase_c.sh"
check_file "$SOURCE_REPO/scripts/run_phase_a.sh"
check_file "$SOURCE_REPO/scripts/run_phase_b.sh"
check_file "$SOURCE_REPO/scripts/run_phase_d.sh"
echo "" | tee -a "$VALIDATION_LOG"

# Check Validation Scripts
echo "Checking Validation Scripts..." | tee -a "$VALIDATION_LOG"
check_file "$SOURCE_REPO/scripts/validate_phase_c.sh"
check_file "$SOURCE_REPO/scripts/validate_phase_a.sh"
check_file "$SOURCE_REPO/scripts/validate_phase_b.sh"
check_file "$SOURCE_REPO/scripts/validate_phase_d.sh"
echo "" | tee -a "$VALIDATION_LOG"

# Check Python Source Files
echo "Checking Python Source Files..." | tee -a "$VALIDATION_LOG"
check_file "$SOURCE_REPO/src/phase_c/extract_empirical_covariance.py"
check_file "$SOURCE_REPO/src/phase_a/calibrate_anchor_tensors.py"
check_file "$SOURCE_REPO/src/phase_b/systematic_grid_epistemic_penalty.py"
check_file "$SOURCE_REPO/src/phase_d/achieve_100pct_resolution.py"
echo "" | tee -a "$VALIDATION_LOG"

# Check Configuration and Utilities
echo "Checking Configuration..." | tee -a "$VALIDATION_LOG"
check_file "$SOURCE_REPO/src/config.py"
check_file "$SOURCE_REPO/src/utils.py"
echo "" | tee -a "$VALIDATION_LOG"

# Check Data Files (CORRECTED PATH)
echo "Checking Data Files..." | tee -a "$VALIDATION_LOG"
check_file "$SOURCE_REPO/data/vizier_data/J_ApJ_826_56_table3.csv"
check_file "$SOURCE_REPO/data/riess2022_covmat.json"
echo "" | tee -a "$VALIDATION_LOG"

# Check MAST Anchor Data
echo "Checking MAST Anchor Data..." | tee -a "$VALIDATION_LOG"
check_file "$SOURCE_MAST/ngc4258_anchors.json"
check_file "$SOURCE_MAST/lmc_anchors.json"
check_file "$SOURCE_MAST/mw_anchors.json"
check_file "$SOURCE_MAST/m31_anchors.json"
echo "" | tee -a "$VALIDATION_LOG"

# Check Reference Data
echo "Checking Reference Data..." | tee -a "$VALIDATION_LOG"
check_file "$SOURCE_REPO/reference/phase_c_reference.json"
check_file "$SOURCE_REPO/reference/phase_a_reference.json"
check_file "$SOURCE_REPO/reference/phase_b_reference.json"
check_file "$SOURCE_REPO/reference/phase_d_reference.json"
echo "" | tee -a "$VALIDATION_LOG"

# Check Documentation
echo "Checking Documentation..." | tee -a "$VALIDATION_LOG"
check_file "$SOURCE_REPO/docs/ANSWER_DATA_FLOW.md"
check_file "$SOURCE_REPO/docs/OBSERVER_TENSOR_LEVELS.md"
echo "" | tee -a "$VALIDATION_LOG"

# Summary
echo "=======================================================================" | tee -a "$VALIDATION_LOG"
echo "Validation Summary" | tee -a "$VALIDATION_LOG"
echo "=======================================================================" | tee -a "$VALIDATION_LOG"
echo "Files Found: $FILES_FOUND" | tee -a "$VALIDATION_LOG"
echo "Files Missing: $FILES_MISSING" | tee -a "$VALIDATION_LOG"
echo "" | tee -a "$VALIDATION_LOG"

if [ $FILES_MISSING -eq 0 ]; then
    echo "✅ PRE-FLIGHT VALIDATION PASSED" | tee -a "$VALIDATION_LOG"
    echo "All required files present. Safe to proceed with consolidation." | tee -a "$VALIDATION_LOG"
    exit 0
else
    echo "❌ PRE-FLIGHT VALIDATION FAILED" | tee -a "$VALIDATION_LOG"
    echo "Missing files must be located before consolidation." | tee -a "$VALIDATION_LOG"
    exit 1
fi
