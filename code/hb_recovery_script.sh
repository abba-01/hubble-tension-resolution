#!/bin/bash
# HB (Hubble Blunder) Recovery Script
# Purpose: Find what contamination caused 97.4% instead of 100% convergence
# Date: October 15, 2025

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/run/media/root/OP01/hb_recovery_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

LOG_FILE="$OUTPUT_DIR/recovery_log.txt"

echo "======================================================================" | tee "$LOG_FILE"
echo "HB RECOVERY - Finding the Contamination" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ======================================================================
# STEP 1: STRUCTURAL COMPARISON
# ======================================================================
echo -e "${BLUE}STEP 1: Structural Comparison${NC}" | tee -a "$LOG_FILE"
echo "Generating directory trees..." | tee -a "$LOG_FILE"

# Original repo tree
cd /run/media/root/OP01/got/hubble
tree -L 3 -a > "$OUTPUT_DIR/original_tree.txt" 2>&1 || {
    echo "  ⚠ tree command not available, using find instead" | tee -a "$LOG_FILE"
    find . -maxdepth 3 | sort > "$OUTPUT_DIR/original_tree.txt"
}
echo "  ✓ Original repo tree: $OUTPUT_DIR/original_tree.txt" | tee -a "$LOG_FILE"

# New clean repo tree
cd /run/media/root/OP01/hubble-tension-resolution
tree -L 3 -a > "$OUTPUT_DIR/clean_tree.txt" 2>&1 || {
    find . -maxdepth 3 | sort > "$OUTPUT_DIR/clean_tree.txt"
}
echo "  ✓ Clean repo tree: $OUTPUT_DIR/clean_tree.txt" | tee -a "$LOG_FILE"

# Compare structures
diff "$OUTPUT_DIR/original_tree.txt" "$OUTPUT_DIR/clean_tree.txt" > "$OUTPUT_DIR/structural_diff.txt" || true
echo "  ✓ Structural diff: $OUTPUT_DIR/structural_diff.txt" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ======================================================================
# STEP 2: CODE COMPARISON - FIND LAYER 6 / ENTROPY CONTAMINATION
# ======================================================================
echo -e "${BLUE}STEP 2: Code Comparison - Finding Layer 6/Entropy Contamination${NC}" | tee -a "$LOG_FILE"

# Search for Layer 6 or entropy references in clean repo
echo "Searching for 'Layer 6' or 'entropy' in clean repo..." | tee -a "$LOG_FILE"
cd /run/media/root/OP01/hubble-tension-resolution
grep -r "layer.*6\|Layer 6\|entropy\|ENTROPY" --include="*.py" . > "$OUTPUT_DIR/contamination_keywords.txt" 2>&1 || true

if [ -s "$OUTPUT_DIR/contamination_keywords.txt" ]; then
    echo -e "  ${RED}✗ FOUND Layer 6/entropy references!${NC}" | tee -a "$LOG_FILE"
    echo "  See: $OUTPUT_DIR/contamination_keywords.txt" | tee -a "$LOG_FILE"
    head -20 "$OUTPUT_DIR/contamination_keywords.txt" | tee -a "$LOG_FILE"
else
    echo -e "  ${GREEN}✓ No obvious Layer 6/entropy keywords found${NC}" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"

# ======================================================================
# STEP 3: PHASE D SPECIFIC COMPARISON
# ======================================================================
echo -e "${BLUE}STEP 3: Phase D Specific Comparison${NC}" | tee -a "$LOG_FILE"

# Check if Phase D files exist
ORIGINAL_PHASE_D="/run/media/root/OP01/got/hubble/code/achieve_100pct_resolution.py"
CLEAN_PHASE_D="/run/media/root/OP01/hubble-tension-resolution/src/phase_d/achieve_100pct_resolution.py"

if [ -f "$ORIGINAL_PHASE_D" ] && [ -f "$CLEAN_PHASE_D" ]; then
    echo "Comparing Phase D implementation..." | tee -a "$LOG_FILE"
    diff "$ORIGINAL_PHASE_D" "$CLEAN_PHASE_D" > "$OUTPUT_DIR/phase_d_diff.txt" || true
    
    if [ -s "$OUTPUT_DIR/phase_d_diff.txt" ]; then
        echo -e "  ${RED}✗ DIFFERENCES FOUND in Phase D!${NC}" | tee -a "$LOG_FILE"
        echo "  See: $OUTPUT_DIR/phase_d_diff.txt" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        echo "  First 50 lines of diff:" | tee -a "$LOG_FILE"
        head -50 "$OUTPUT_DIR/phase_d_diff.txt" | tee -a "$LOG_FILE"
    else
        echo -e "  ${GREEN}✓ Phase D files are identical${NC}" | tee -a "$LOG_FILE"
    fi
else
    echo -e "  ${YELLOW}⚠ Phase D files not found in expected locations${NC}" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"

# ======================================================================
# STEP 4: RESULTS COMPARISON
# ======================================================================
echo -e "${BLUE}STEP 4: Results Comparison${NC}" | tee -a "$LOG_FILE"

ORIGINAL_RESULTS="/run/media/root/OP01/got/hubble/results/resolution_100pct_mcmc.json"
CLEAN_RESULTS="/run/media/root/OP01/hubble-tension-resolution/results/phase_d/resolution_100pct_mcmc.json"

echo "Comparing result files..." | tee -a "$LOG_FILE"

# Check original results
if [ -f "$ORIGINAL_RESULTS" ]; then
    echo "  Original results:" | tee -a "$LOG_FILE"
    grep -E "reduction_percent|H0_merged|gap_remaining" "$ORIGINAL_RESULTS" | tee -a "$LOG_FILE"
    cp "$ORIGINAL_RESULTS" "$OUTPUT_DIR/original_results.json"
else
    echo -e "  ${YELLOW}⚠ Original results not found${NC}" | tee -a "$LOG_FILE"
fi

# Check clean results
if [ -f "$CLEAN_RESULTS" ]; then
    echo "  Clean repo results:" | tee -a "$LOG_FILE"
    grep -E "reduction_percent|H0_merged|gap_remaining" "$CLEAN_RESULTS" | tee -a "$LOG_FILE"
    cp "$CLEAN_RESULTS" "$OUTPUT_DIR/clean_results.json"
else
    echo -e "  ${YELLOW}⚠ Clean results not found${NC}" | tee -a "$LOG_FILE"
fi

# Compare if both exist
if [ -f "$ORIGINAL_RESULTS" ] && [ -f "$CLEAN_RESULTS" ]; then
    diff "$ORIGINAL_RESULTS" "$CLEAN_RESULTS" > "$OUTPUT_DIR/results_diff.txt" || true
    if [ -s "$OUTPUT_DIR/results_diff.txt" ]; then
        echo -e "  ${RED}✗ Results differ!${NC}" | tee -a "$LOG_FILE"
        echo "  See: $OUTPUT_DIR/results_diff.txt" | tee -a "$LOG_FILE"
    else
        echo -e "  ${GREEN}✓ Results are identical${NC}" | tee -a "$LOG_FILE"
    fi
fi
echo "" | tee -a "$LOG_FILE"

# ======================================================================
# STEP 5: SEARCH FOR FULL CONVERGENCE FILE
# ======================================================================
echo -e "${BLUE}STEP 5: Searching for Full Convergence (100%) File${NC}" | tee -a "$LOG_FILE"

cd /run/media/root/OP01/got

echo "Searching for files with full_concordance: true..." | tee -a "$LOG_FILE"
find . -name "*.json" -exec grep -l '"full_concordance".*true' {} \; > "$OUTPUT_DIR/full_concordance_files.txt" 2>&1 || true

echo "Searching for 100% resolution claims..." | tee -a "$LOG_FILE"
grep -r "100.0.*convergence\|100%.*resolution\|100.*concordance" \
    --include="*.json" --include="*.txt" --include="*.md" \
    hubble*/ > "$OUTPUT_DIR/100_percent_claims.txt" 2>&1 || true

if [ -s "$OUTPUT_DIR/full_concordance_files.txt" ]; then
    echo -e "  ${GREEN}✓ Found files claiming full concordance:${NC}" | tee -a "$LOG_FILE"
    cat "$OUTPUT_DIR/full_concordance_files.txt" | tee -a "$LOG_FILE"
else
    echo -e "  ${YELLOW}⚠ No files found with full_concordance: true${NC}" | tee -a "$LOG_FILE"
fi

if [ -s "$OUTPUT_DIR/100_percent_claims.txt" ]; then
    echo -e "  ${GREEN}✓ Found 100% claims in:${NC}" | tee -a "$LOG_FILE"
    head -20 "$OUTPUT_DIR/100_percent_claims.txt" | tee -a "$LOG_FILE"
else
    echo -e "  ${YELLOW}⚠ No 100% convergence claims found${NC}" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"

# ======================================================================
# STEP 6: DATA FILE COMPARISON
# ======================================================================
echo -e "${BLUE}STEP 6: Data File Comparison${NC}" | tee -a "$LOG_FILE"

echo "Checking MCMC samples..." | tee -a "$LOG_FILE"

# Original MCMC
if [ -d "/run/media/root/OP01/got/hubble/data/mast" ]; then
    ls -lh /run/media/root/OP01/got/hubble/data/mast/mcmc_samples_*.npy > "$OUTPUT_DIR/original_mcmc_files.txt" 2>&1 || true
    echo "  Original MCMC files listed in: $OUTPUT_DIR/original_mcmc_files.txt" | tee -a "$LOG_FILE"
fi

# Clean MCMC  
if [ -d "/run/media/root/OP01/hubble-tension-resolution/data/mast_anchors" ]; then
    ls -lh /run/media/root/OP01/hubble-tension-resolution/data/mast_anchors/mcmc_samples_*.npy > "$OUTPUT_DIR/clean_mcmc_files.txt" 2>&1 || true
    echo "  Clean repo MCMC files listed in: $OUTPUT_DIR/clean_mcmc_files.txt" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"

# ======================================================================
# STEP 7: HUNT FOR ORPHANED/MISSING FILES
# ======================================================================
echo -e "${BLUE}STEP 7: Hunt for Orphaned/Missing Files${NC}" | tee -a "$LOG_FILE"

cd /run/media/root/OP01/got/hubble

echo "Searching for Phase D dependencies..." | tee -a "$LOG_FILE"
grep -h "open\|load\|read" code/achieve_100pct_resolution.py 2>/dev/null | \
    grep -v "^#" | grep -E "\.(json|csv|npy|dat)" > "$OUTPUT_DIR/phase_d_dependencies.txt" || true

echo "Files referenced by Phase D:" | tee -a "$LOG_FILE"
cat "$OUTPUT_DIR/phase_d_dependencies.txt" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Check for recently modified files that might be missing
echo "Recently modified JSON files in original repo:" | tee -a "$LOG_FILE"
find . -name "*.json" -mtime -30 -ls > "$OUTPUT_DIR/recent_json_files.txt" 2>&1 || true
head -20 "$OUTPUT_DIR/recent_json_files.txt" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ======================================================================
# STEP 8: CHECK FOR TWO VERSIONS OF CORRECTED_RESULTS
# ======================================================================
echo -e "${BLUE}STEP 8: Checking CORRECTED_RESULTS_32BIT.json Versions${NC}" | tee -a "$LOG_FILE"

VERSION1="/run/media/root/OP01/got/hubble/CORRECTED_RESULTS_32BIT.json"
VERSION2="/run/media/root/OP01/got/hubble/02_hubble_analysis/CORRECTED_RESULTS_32BIT.json"

if [ -f "$VERSION1" ]; then
    echo "  Version 1 (root):" | tee -a "$LOG_FILE"
    grep -E "full_concordance|gap_remaining|resolution" "$VERSION1" | tee -a "$LOG_FILE"
    cp "$VERSION1" "$OUTPUT_DIR/CORRECTED_RESULTS_v1.json"
fi

if [ -f "$VERSION2" ]; then
    echo "  Version 2 (02_hubble_analysis):" | tee -a "$LOG_FILE"
    grep -E "full_concordance|gap_remaining|resolution" "$VERSION2" | tee -a "$LOG_FILE"
    cp "$VERSION2" "$OUTPUT_DIR/CORRECTED_RESULTS_v2.json"
fi

if [ -f "$VERSION1" ] && [ -f "$VERSION2" ]; then
    diff "$VERSION1" "$VERSION2" > "$OUTPUT_DIR/corrected_results_versions_diff.txt" || true
    if [ -s "$OUTPUT_DIR/corrected_results_versions_diff.txt" ]; then
        echo -e "  ${RED}✗ Two different versions exist!${NC}" | tee -a "$LOG_FILE"
        echo "  See: $OUTPUT_DIR/corrected_results_versions_diff.txt" | tee -a "$LOG_FILE"
    else
        echo -e "  ${GREEN}✓ Both versions are identical${NC}" | tee -a "$LOG_FILE"
    fi
fi
echo "" | tee -a "$LOG_FILE"

# ======================================================================
# SUMMARY
# ======================================================================
echo "======================================================================" | tee -a "$LOG_FILE"
echo "RECOVERY ANALYSIS COMPLETE" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
echo "Completed: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "All output files saved to: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Key files to review:" | tee -a "$LOG_FILE"
echo "  1. $OUTPUT_DIR/phase_d_diff.txt - Phase D code differences" | tee -a "$LOG_FILE"
echo "  2. $OUTPUT_DIR/contamination_keywords.txt - Layer 6/entropy references" | tee -a "$LOG_FILE"
echo "  3. $OUTPUT_DIR/results_diff.txt - Results comparison" | tee -a "$LOG_FILE"
echo "  4. $OUTPUT_DIR/full_concordance_files.txt - Files claiming 100%" | tee -a "$LOG_FILE"
echo "  5. $OUTPUT_DIR/structural_diff.txt - Directory structure differences" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Check for critical findings
CRITICAL_FINDINGS=0

if [ -s "$OUTPUT_DIR/phase_d_diff.txt" ]; then
    echo -e "${RED}CRITICAL: Phase D code differs between repos${NC}" | tee -a "$LOG_FILE"
    ((CRITICAL_FINDINGS++))
fi

if [ -s "$OUTPUT_DIR/contamination_keywords.txt" ]; then
    echo -e "${RED}CRITICAL: Layer 6/entropy contamination found${NC}" | tee -a "$LOG_FILE"
    ((CRITICAL_FINDINGS++))
fi

if [ -s "$OUTPUT_DIR/corrected_results_versions_diff.txt" ]; then
    echo -e "${YELLOW}WARNING: Multiple versions of CORRECTED_RESULTS exist${NC}" | tee -a "$LOG_FILE"
    ((CRITICAL_FINDINGS++))
fi

if [ $CRITICAL_FINDINGS -eq 0 ]; then
    echo -e "${GREEN}No obvious contamination found. The 97.4% may be the actual result.${NC}" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "Next steps:" | tee -a "$LOG_FILE"
echo "  1. Review the diff files to identify specific changes" | tee -a "$LOG_FILE"
echo "  2. If contamination found, revert changes in clean repo" | tee -a "$LOG_FILE"
echo "  3. Re-run Phase D to verify 100% convergence" | tee -a "$LOG_FILE"
echo "  4. If no contamination found, accept 97.4% as correct result" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Full log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
