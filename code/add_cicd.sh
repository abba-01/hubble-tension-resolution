#!/bin/bash
# CI/CD Integration Script
# Purpose: Add repository manifest, provenance YAML, and CI/CD automation
# Date: October 14, 2025

set -e
umask 022

TARGET_REPO="/run/media/root/OP01/hubble_tension_clean"
TIMESTAMP=$(date +%Y%m%d_%H%M%S 2>/dev/null || gdate +%Y%m%d_%H%M%S)
LOG_FILE="add_cicd_${TIMESTAMP}.log"

echo "=======================================================================" | tee "$LOG_FILE"
echo "Adding CI/CD Integration" | tee -a "$LOG_FILE"
echo "=======================================================================" | tee -a "$LOG_FILE"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

cd "$TARGET_REPO"

# ======================================================================
# 1. Repository Manifest (visual tree structure)
# ======================================================================
echo "1. Generating repository manifest..." | tee -a "$LOG_FILE"
MANIFEST_FILE="SAID/audit_logs/repo_manifest_${TIMESTAMP}.txt"

if command -v tree >/dev/null 2>&1; then
    tree -I '.git|__pycache__' "$TARGET_REPO" > "$MANIFEST_FILE"
else
    find "$TARGET_REPO" -not -path '*/\.git/*' -not -path '*/__pycache__/*' | sort > "$MANIFEST_FILE"
fi

echo "  ✓ Manifest: $MANIFEST_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ======================================================================
# 2. Provenance YAML (Enhanced)
# ======================================================================
echo "2. Generating provenance metadata..." | tee -a "$LOG_FILE"

PROVENANCE_FILE="SAID/provenance/consolidation_provenance_${TIMESTAMP}.yaml"
cat > "$PROVENANCE_FILE" << EOFPROV
# Consolidation Provenance Metadata
# Generated: $(date '+%Y-%m-%d %H:%M:%S')

consolidation:
  timestamp: $(date -Iseconds)
  hostname: $(hostname)
  user: $(whoami)

credits:
  framework_design: Claude (Sonnet 4.5)
  filesystem_verification: ClaudeCode
  validation_review: GPT-5
  orchestration: Human operator

environment:
  os: $(uname -s)
  architecture: $(uname -m)
  kernel: $(uname -r)

software_versions:
  bash: $(bash --version | head -n1)
  python: $(python3 --version 2>&1)
  git: $(git --version)

repository:
  source: /run/media/root/OP01/got/hubble
  mast_data: /run/media/root/OP01/got/hubble_data/mast_anchors
  target: $TARGET_REPO

file_counts:
  scripts: $(find scripts/ -name "*.sh" 2>/dev/null | wc -l)
  python_modules: $(find src/ -name "*.py" 2>/dev/null | wc -l)
  data_files: $(find data/ -type f 2>/dev/null | wc -l)
  reference_files: $(find reference/ -name "*.json" 2>/dev/null | wc -l)

checksums:
  source_checksums: SAID/verification/checksums_${TIMESTAMP}.txt
  manifest: $MANIFEST_FILE

git:
  commit: $(git rev-parse HEAD 2>/dev/null || echo "N/A")
  branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "N/A")

reproducibility:
  test_script: tests/test_reproducibility.py
  validation_scripts:
    - scripts/validate_phase_c.sh
    - scripts/validate_phase_a.sh
    - scripts/validate_phase_b.sh
    - scripts/validate_phase_d.sh
EOFPROV

echo "  ✓ Provenance: $PROVENANCE_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ======================================================================
# 3. Reproducibility Test
# ======================================================================
echo "3. Creating reproducibility test..." | tee -a "$LOG_FILE"

cat > "tests/test_reproducibility.py" << 'EOFPY'
#!/usr/bin/env python3
"""
Reproducibility Test
Compares pipeline outputs against reference values.
"""

import json
import sys
from pathlib import Path

def compare_phase_output(phase_name, output_file, reference_file, tolerance=1e-6):
    """Compare phase output against reference."""
    print(f"\n{'='*70}")
    print(f"Testing {phase_name}")
    print(f"{'='*70}")

    try:
        with open(output_file) as f:
            output = json.load(f)
        with open(reference_file) as f:
            reference = json.load(f)
    except FileNotFoundError as e:
        print(f"❌ FAIL: {e}")
        return False

    # Compare key values
    mismatches = []
    for key in reference:
        if key not in output:
            mismatches.append(f"Missing key: {key}")
            continue

        ref_val = reference[key]
        out_val = output[key]

        # Numeric comparison
        if isinstance(ref_val, (int, float)) and isinstance(out_val, (int, float)):
            if abs(ref_val - out_val) > tolerance:
                mismatches.append(f"{key}: {out_val} != {ref_val} (diff > {tolerance})")
        # Exact comparison for others
        elif ref_val != out_val:
            mismatches.append(f"{key}: {out_val} != {ref_val}")

    if mismatches:
        print(f"❌ FAIL: {len(mismatches)} mismatches")
        for mismatch in mismatches[:5]:  # Show first 5
            print(f"  - {mismatch}")
        if len(mismatches) > 5:
            print(f"  ... and {len(mismatches) - 5} more")
        return False
    else:
        print(f"✅ PASS: All values match within tolerance")
        return True

def main():
    repo_root = Path(__file__).parent.parent

    tests = [
        ("Phase C",
         repo_root / "results/phase_c/covariance_eigenspectrum.json",
         repo_root / "reference/phase_c_reference.json"),
        ("Phase A",
         repo_root / "results/phase_a/anchor_calibration.json",
         repo_root / "reference/phase_a_reference.json"),
        ("Phase B",
         repo_root / "results/phase_b/epistemic_penalty_grid.json",
         repo_root / "reference/phase_b_reference.json"),
        ("Phase D",
         repo_root / "results/phase_d/final_resolution.json",
         repo_root / "reference/phase_d_reference.json"),
    ]

    results = []
    for phase_name, output_file, reference_file in tests:
        results.append(compare_phase_output(phase_name, output_file, reference_file))

    print(f"\n{'='*70}")
    print(f"Summary: {sum(results)}/{len(results)} tests passed")
    print(f"{'='*70}\n")

    sys.exit(0 if all(results) else 1)

if __name__ == "__main__":
    main()
EOFPY

chmod +x tests/test_reproducibility.py
echo "  ✓ Test script: tests/test_reproducibility.py" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ======================================================================
# 4. GitHub Actions Workflow
# ======================================================================
echo "4. Creating GitHub Actions workflow..." | tee -a "$LOG_FILE"

mkdir -p .github/workflows
cat > .github/workflows/reproducibility.yml << 'EOFGH'
name: Reproducibility Check

on:
  push:
    branches: [ main, master, develop ]
  pull_request:
    branches: [ main, master, develop ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install numpy scipy

    - name: Run reproducibility test
      run: python tests/test_reproducibility.py
EOFGH

echo "  ✓ GitHub Actions: .github/workflows/reproducibility.yml" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ======================================================================
# 5. GitLab CI Configuration
# ======================================================================
echo "5. Creating GitLab CI config..." | tee -a "$LOG_FILE"

cat > .gitlab-ci.yml << 'EOFGL'
image: python:3.9

stages:
  - test

reproducibility_test:
  stage: test
  script:
    - pip install numpy scipy
    - python tests/test_reproducibility.py
  only:
    - main
    - master
    - develop
EOFGL

echo "  ✓ GitLab CI: .gitlab-ci.yml" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ======================================================================
# 6. Makefile (WITH ACTUAL TAB CHARACTERS)
# ======================================================================
echo "6. Creating Makefile with actual TAB characters..." | tee -a "$LOG_FILE"

# Create Makefile with ACTUAL tab characters using printf
{
    echo ".PHONY: verify clean help"
    echo ""
    echo "help:"
    printf '\t@echo "Available targets:"\n'
    printf '\t@echo "  make verify  - Run reproducibility test"\n'
    printf '\t@echo "  make clean   - Clean temporary files"\n'
    echo ""
    echo "verify:"
    printf '\t@echo "Running reproducibility test..."\n'
    printf '\t@python tests/test_reproducibility.py\n'
    echo ""
    echo "clean:"
    printf '\t@echo "Cleaning temporary files..."\n'
    printf '\t@find . -type f -name "*.pyc" -delete\n'
    printf '\t@find . -type d -name "__pycache__" -delete\n'
    printf '\t@echo "Done."\n'
} > Makefile

echo "  ✓ Makefile created (with actual TAB characters)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ======================================================================
# 7. Update README with CI/CD info
# ======================================================================
echo "7. Updating README with CI/CD instructions..." | tee -a "$LOG_FILE"

cat >> README.md << 'EOFREADME'

## Reproducibility

### Manual Verification
```bash
# Run reproducibility test
python tests/test_reproducibility.py

# Or use make
make verify
```

### Automated CI/CD
- **GitHub Actions**: Runs on every push to main/master/develop
- **GitLab CI**: Runs on merge requests
- Both platforms automatically verify reproducibility

### Provenance
Complete audit trail available in:
- `SAID/audit_logs/` - Repository manifests
- `SAID/verification/` - File checksums
- `SAID/provenance/` - Consolidation metadata

EOFREADME

echo "  ✓ README.md updated" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ======================================================================
# 8. Commit all CI/CD additions
# ======================================================================
echo "8. Committing CI/CD additions..." | tee -a "$LOG_FILE"

git add .
git commit -m "Add CI/CD integration and reproducibility testing

- Repository manifest for structure snapshot
- Provenance YAML with environment metadata
- Reproducibility test comparing outputs to references
- GitHub Actions and GitLab CI workflows
- Makefile for easy verification (with actual TAB characters)
- Updated README with testing instructions

Generated: $(date '+%Y-%m-%d %H:%M:%S')"

echo "  ✓ Changes committed" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ======================================================================
# Summary
# ======================================================================
echo "=======================================================================" | tee -a "$LOG_FILE"
echo "CI/CD Integration Complete" | tee -a "$LOG_FILE"
echo "=======================================================================" | tee -a "$LOG_FILE"
echo "Completed: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Added:" | tee -a "$LOG_FILE"
echo "  ✓ Repository manifest: $MANIFEST_FILE" | tee -a "$LOG_FILE"
echo "  ✓ Provenance YAML: $PROVENANCE_FILE" | tee -a "$LOG_FILE"
echo "  ✓ Reproducibility test: tests/test_reproducibility.py" | tee -a "$LOG_FILE"
echo "  ✓ GitHub Actions: .github/workflows/reproducibility.yml" | tee -a "$LOG_FILE"
echo "  ✓ GitLab CI: .gitlab-ci.yml" | tee -a "$LOG_FILE"
echo "  ✓ Makefile: Makefile (with REAL tabs)" | tee -a "$LOG_FILE"
echo "  ✓ Updated: README.md" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Next Steps:" | tee -a "$LOG_FILE"
echo "1. Verify Makefile tabs: cat -A Makefile | grep '^I'" | tee -a "$LOG_FILE"
echo "2. Test locally: python tests/test_reproducibility.py" | tee -a "$LOG_FILE"
echo "3. Test with make: make verify" | tee -a "$LOG_FILE"
echo "4. Push to trigger CI/CD: git push origin main" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "✅ Repository is now CI/CD-ready with full audit trail" | tee -a "$LOG_FILE"
