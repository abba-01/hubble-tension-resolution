#!/bin/bash
# Repository Consolidation Script v3 - Production Grade
# Base structure: Claude (Sonnet 4.5)
# Verification & enhancements: ClaudeCode
# Validation & refinements: GPT-5
# Purpose: Migrate verified scripts to clean repository structure

set -e
umask 022

# Cross-platform timestamp functions
get_timestamp() {
    date +"%Y-%m-%d %H:%M:%S" 2>/dev/null || gdate +"%Y-%m-%d %H:%M:%S"
}

get_time() {
    date +%Y%m%d_%H%M%S 2>/dev/null || gdate +%Y%m%d_%H%M%S
}

# Paths
SOURCE_REPO="/run/media/root/OP01/got/hubble"
SOURCE_MAST="/run/media/root/OP01/got/hubble_data/mast_anchors"
TARGET_REPO="/run/media/root/OP01/hubble_tension_clean"
TIMESTAMP=$(get_time)
LOG_FILE="consolidation_${TIMESTAMP}.log"

# Redirect all errors to log
exec 2>>"$LOG_FILE"

# Graceful interrupt handling
trap 'echo "✗ Interrupted at $(get_timestamp)" | tee -a "$LOG_FILE"; exit 130' INT

# ======================================================================
# Pre-Flight Checks (GPT-5 enhancement)
# ======================================================================
echo "=======================================================================" | tee "$LOG_FILE"
echo "Repository Consolidation Script v3" | tee -a "$LOG_FILE"
echo "=======================================================================" | tee -a "$LOG_FILE"
echo "Started: $(get_timestamp)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "STEP 0: Pre-Flight Checks" | tee -a "$LOG_FILE"
if [ ! -d "$SOURCE_REPO" ]; then
    echo "✗ ERROR: Source repository not found: $SOURCE_REPO" | tee -a "$LOG_FILE"
    exit 1
fi

if [ ! -d "$SOURCE_MAST" ]; then
    echo "✗ ERROR: MAST data directory not found: $SOURCE_MAST" | tee -a "$LOG_FILE"
    exit 1
fi

echo "  ✓ Source repository: $SOURCE_REPO" | tee -a "$LOG_FILE"
echo "  ✓ MAST data: $SOURCE_MAST" | tee -a "$LOG_FILE"
echo "  ✓ Target: $TARGET_REPO" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Counters
SCRIPTS_FOUND=0
SCRIPTS_MISSING=0
FILES_COPIED=0

# ======================================================================
# STEP 1: Create Directory Structure
# ======================================================================
echo "STEP 1: Creating directory structure..." | tee -a "$LOG_FILE"
mkdir -p "$TARGET_REPO"/{scripts,src/{phase_c,phase_a,phase_b,phase_d},data/{vizier_data,mast_anchors},results/{phase_c,phase_a,phase_b,phase_d},reference,tests,docs,SAID/{audit_logs,verification,provenance}}
echo "  ✓ Directory structure created" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ======================================================================
# STEP 2: Copy Phase Scripts
# ======================================================================
echo "STEP 2: Copying phase execution scripts..." | tee -a "$LOG_FILE"
for script in run_phase_c.sh run_phase_a.sh run_phase_b.sh run_phase_d.sh; do
    if [ -f "$SOURCE_REPO/scripts/$script" ]; then
        cp -p "$SOURCE_REPO/scripts/$script" "$TARGET_REPO/scripts/"
        echo "  ✓ Copied: scripts/$script" | tee -a "$LOG_FILE"
        ((SCRIPTS_FOUND++))
        ((FILES_COPIED++))
    else
        echo "  ✗ Missing: scripts/$script" | tee -a "$LOG_FILE"
        ((SCRIPTS_MISSING++))
    fi
done
echo "" | tee -a "$LOG_FILE"

# ======================================================================
# STEP 3: Copy Validation Scripts
# ======================================================================
echo "STEP 3: Copying validation scripts..." | tee -a "$LOG_FILE"
for script in validate_phase_c.sh validate_phase_a.sh validate_phase_b.sh validate_phase_d.sh; do
    if [ -f "$SOURCE_REPO/scripts/$script" ]; then
        cp -p "$SOURCE_REPO/scripts/$script" "$TARGET_REPO/scripts/"
        echo "  ✓ Copied: scripts/$script" | tee -a "$LOG_FILE"
        ((SCRIPTS_FOUND++))
        ((FILES_COPIED++))
    else
        echo "  ✗ Missing: scripts/$script" | tee -a "$LOG_FILE"
        ((SCRIPTS_MISSING++))
    fi
done
echo "" | tee -a "$LOG_FILE"

# ======================================================================
# STEP 4-7: Copy Python Source Files
# ======================================================================
echo "STEP 4: Copying Phase C source..." | tee -a "$LOG_FILE"
if [ -f "$SOURCE_REPO/src/phase_c/extract_empirical_covariance.py" ]; then
    cp -p "$SOURCE_REPO/src/phase_c/extract_empirical_covariance.py" "$TARGET_REPO/src/phase_c/"
    echo "  ✓ Copied: src/phase_c/extract_empirical_covariance.py" | tee -a "$LOG_FILE"
    ((FILES_COPIED++))
fi
echo "" | tee -a "$LOG_FILE"

echo "STEP 5: Copying Phase A source..." | tee -a "$LOG_FILE"
if [ -f "$SOURCE_REPO/src/phase_a/calibrate_anchor_tensors.py" ]; then
    cp -p "$SOURCE_REPO/src/phase_a/calibrate_anchor_tensors.py" "$TARGET_REPO/src/phase_a/"
    echo "  ✓ Copied: src/phase_a/calibrate_anchor_tensors.py" | tee -a "$LOG_FILE"
    ((FILES_COPIED++))
fi
echo "" | tee -a "$LOG_FILE"

echo "STEP 6: Copying Phase B source..." | tee -a "$LOG_FILE"
if [ -f "$SOURCE_REPO/src/phase_b/systematic_grid_epistemic_penalty.py" ]; then
    cp -p "$SOURCE_REPO/src/phase_b/systematic_grid_epistemic_penalty.py" "$TARGET_REPO/src/phase_b/"
    echo "  ✓ Copied: src/phase_b/systematic_grid_epistemic_penalty.py" | tee -a "$LOG_FILE"
    ((FILES_COPIED++))
fi
echo "" | tee -a "$LOG_FILE"

echo "STEP 7: Copying Phase D source..." | tee -a "$LOG_FILE"
if [ -f "$SOURCE_REPO/src/phase_d/achieve_100pct_resolution.py" ]; then
    cp -p "$SOURCE_REPO/src/phase_d/achieve_100pct_resolution.py" "$TARGET_REPO/src/phase_d/"
    echo "  ✓ Copied: src/phase_d/achieve_100pct_resolution.py" | tee -a "$LOG_FILE"
    ((FILES_COPIED++))
fi
echo "" | tee -a "$LOG_FILE"

# ======================================================================
# STEP 8: Copy Configuration Files
# ======================================================================
echo "STEP 8: Copying configuration files..." | tee -a "$LOG_FILE"
if [ -f "$SOURCE_REPO/src/config.py" ]; then
    cp -p "$SOURCE_REPO/src/config.py" "$TARGET_REPO/src/"
    echo "  ✓ Copied: src/config.py" | tee -a "$LOG_FILE"
    ((FILES_COPIED++))
fi
if [ -f "$SOURCE_REPO/src/utils.py" ]; then
    cp -p "$SOURCE_REPO/src/utils.py" "$TARGET_REPO/src/"
    echo "  ✓ Copied: src/utils.py" | tee -a "$LOG_FILE"
    ((FILES_COPIED++))
fi
echo "" | tee -a "$LOG_FILE"

# ======================================================================
# STEP 9: Copy Data Files (CORRECTED PATH)
# ======================================================================
echo "STEP 9: Copying systematic grid data..." | tee -a "$LOG_FILE"
if [ -f "$SOURCE_REPO/data/vizier_data/J_ApJ_826_56_table3.csv" ]; then
    cp -p "$SOURCE_REPO/data/vizier_data/J_ApJ_826_56_table3.csv" "$TARGET_REPO/data/vizier_data/"
    echo "  ✓ Copied: data/vizier_data/J_ApJ_826_56_table3.csv" | tee -a "$LOG_FILE"
    ((FILES_COPIED++))
else
    echo "  ✗ Missing: data/vizier_data/J_ApJ_826_56_table3.csv" | tee -a "$LOG_FILE"
fi

if [ -f "$SOURCE_REPO/data/riess2022_covmat.json" ]; then
    cp -p "$SOURCE_REPO/data/riess2022_covmat.json" "$TARGET_REPO/data/"
    echo "  ✓ Copied: data/riess2022_covmat.json" | tee -a "$LOG_FILE"
    ((FILES_COPIED++))
fi
echo "" | tee -a "$LOG_FILE"

# ======================================================================
# STEP 10: Copy MAST Anchor Data
# ======================================================================
echo "STEP 10: Copying MAST anchor data..." | tee -a "$LOG_FILE"
for anchor in ngc4258_anchors.json lmc_anchors.json mw_anchors.json m31_anchors.json; do
    if [ -f "$SOURCE_MAST/$anchor" ]; then
        cp -p "$SOURCE_MAST/$anchor" "$TARGET_REPO/data/mast_anchors/"
        echo "  ✓ Copied: data/mast_anchors/$anchor" | tee -a "$LOG_FILE"
        ((FILES_COPIED++))
    fi
done
echo "" | tee -a "$LOG_FILE"

# ======================================================================
# STEP 11: Copy Reference Data
# ======================================================================
echo "STEP 11: Copying reference data..." | tee -a "$LOG_FILE"
for ref in phase_c_reference.json phase_a_reference.json phase_b_reference.json phase_d_reference.json; do
    if [ -f "$SOURCE_REPO/reference/$ref" ]; then
        cp -p "$SOURCE_REPO/reference/$ref" "$TARGET_REPO/reference/"
        echo "  ✓ Copied: reference/$ref" | tee -a "$LOG_FILE"
        ((FILES_COPIED++))
    fi
done
echo "" | tee -a "$LOG_FILE"

# ======================================================================
# STEP 12: Copy Documentation
# ======================================================================
echo "STEP 12: Copying documentation..." | tee -a "$LOG_FILE"
if [ -f "$SOURCE_REPO/docs/ANSWER_DATA_FLOW.md" ]; then
    cp -p "$SOURCE_REPO/docs/ANSWER_DATA_FLOW.md" "$TARGET_REPO/docs/"
    echo "  ✓ Copied: docs/ANSWER_DATA_FLOW.md" | tee -a "$LOG_FILE"
    ((FILES_COPIED++))
fi
if [ -f "$SOURCE_REPO/docs/OBSERVER_TENSOR_LEVELS.md" ]; then
    cp -p "$SOURCE_REPO/docs/OBSERVER_TENSOR_LEVELS.md" "$TARGET_REPO/docs/"
    echo "  ✓ Copied: docs/OBSERVER_TENSOR_LEVELS.md" | tee -a "$LOG_FILE"
    ((FILES_COPIED++))
fi
echo "" | tee -a "$LOG_FILE"

# ======================================================================
# STEP 13: Create README
# ======================================================================
echo "STEP 13: Creating README..." | tee -a "$LOG_FILE"
cat > "$TARGET_REPO/README.md" << 'EOFREADME'
# Hubble Tension Resolution via Epistemic Merging

## Overview
This repository implements a novel framework for resolving the Hubble tension using epistemic merging of discordant measurements.

## Structure
```
hubble_tension_clean/
├── scripts/          # Execution and validation scripts
├── src/              # Python source code (phases A-D)
├── data/             # Input data (systematic grid, MAST anchors)
├── results/          # Output data (phase results)
├── reference/        # Reference values for validation
├── tests/            # Reproducibility tests
├── docs/             # Documentation
└── SAID/             # Audit trail and provenance
```

## Quick Start
```bash
# Run complete pipeline
bash scripts/run_phase_c.sh
bash scripts/run_phase_a.sh
bash scripts/run_phase_b.sh
bash scripts/run_phase_d.sh

# Validate results
bash scripts/validate_phase_c.sh
bash scripts/validate_phase_a.sh
bash scripts/validate_phase_b.sh
bash scripts/validate_phase_d.sh
```

## Credits
- Framework: Claude (Sonnet 4.5)
- Verification: ClaudeCode
- Validation: GPT-5

## Consolidation
Consolidated: $(get_timestamp)
EOFREADME
echo "  ✓ README.md created" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ======================================================================
# STEP 14: Initialize Git (GPT-5 enhancement - localized config)
# ======================================================================
echo "STEP 14: Initializing git repository..." | tee -a "$LOG_FILE"
cd "$TARGET_REPO"
git init
git config user.name "Consolidation Bot"
git config user.email "consolidation@hubble.local"
git add .

COMMIT_MSG="Initial consolidation $(date +"%Y-%m-%d")

Consolidation completed at $(get_timestamp)

Credits:
  - Base: Claude (Sonnet 4.5)
  - Verification: ClaudeCode
  - Validation: GPT-5

Scripts found: $SCRIPTS_FOUND | missing: $SCRIPTS_MISSING
Files copied: $FILES_COPIED"

git commit -m "$COMMIT_MSG"
echo "  ✓ Git repository initialized" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ======================================================================
# STEP 15: Generate SOURCE Checksums for Provenance (CORRECTED)
# ======================================================================
echo "STEP 15: Generating source file checksums for provenance..." | tee -a "$LOG_FILE"
CHECKSUM_FILE="SAID/verification/checksums_${TIMESTAMP}.txt"

# Checksum SOURCE files (for provenance, not circular target check)
echo "# Source File Checksums - Generated $(get_timestamp)" > "$CHECKSUM_FILE"
echo "# Purpose: Record state of source files at consolidation time" >> "$CHECKSUM_FILE"
echo "" >> "$CHECKSUM_FILE"

find "$SOURCE_REPO" -type f \( -name "*.py" -o -name "*.sh" -o -name "*.json" -o -name "*.csv" \) \
    -print0 | xargs -0 sha256sum >> "$CHECKSUM_FILE" 2>/dev/null || true

find "$SOURCE_MAST" -type f -name "*.json" \
    -print0 | xargs -0 sha256sum >> "$CHECKSUM_FILE" 2>/dev/null || true

echo "  ✓ Source checksums: $CHECKSUM_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ======================================================================
# STEP 16: Create Distribution Archive (GPT-5 enhancement)
# ======================================================================
echo "STEP 16: Creating distribution archive..." | tee -a "$LOG_FILE"
cd /run/media/root/OP01
ARCHIVE_NAME="hubble_tension_${TIMESTAMP}.tar.gz"
tar -czf "$ARCHIVE_NAME" hubble_tension_clean/
sha256sum "$ARCHIVE_NAME" > "${ARCHIVE_NAME}.sha256"
echo "  ✓ Archive: $ARCHIVE_NAME" | tee -a "$LOG_FILE"
echo "  ✓ Archive checksum: ${ARCHIVE_NAME}.sha256" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ======================================================================
# Summary
# ======================================================================
echo "=======================================================================" | tee -a "$LOG_FILE"
echo "Consolidation Summary" | tee -a "$LOG_FILE"
echo "=======================================================================" | tee -a "$LOG_FILE"
echo "Completed: $(get_timestamp)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Scripts Found: $SCRIPTS_FOUND" | tee -a "$LOG_FILE"
echo "Scripts Missing: $SCRIPTS_MISSING" | tee -a "$LOG_FILE"
echo "Files Copied: $FILES_COPIED" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Repository: $TARGET_REPO" | tee -a "$LOG_FILE"
echo "Archive: $ARCHIVE_NAME" | tee -a "$LOG_FILE"
echo "Checksum: ${ARCHIVE_NAME}.sha256" | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if [ $SCRIPTS_MISSING -eq 0 ]; then
    echo "✅ CONSOLIDATION SUCCESSFUL" | tee -a "$LOG_FILE"
else
    echo "⚠️  CONSOLIDATION COMPLETE WITH WARNINGS" | tee -a "$LOG_FILE"
    echo "Some expected scripts were not found (see log for details)" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "Next Steps:" | tee -a "$LOG_FILE"
echo "1. Verify archive: sha256sum -c ${ARCHIVE_NAME}.sha256" | tee -a "$LOG_FILE"
echo "2. Extract: tar -xzf $ARCHIVE_NAME" | tee -a "$LOG_FILE"
echo "3. Run Phase C: cd hubble_tension_clean && bash scripts/run_phase_c.sh" | tee -a "$LOG_FILE"
