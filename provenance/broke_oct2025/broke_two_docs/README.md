# Hubble Tension Resolution via N/U Algebra

Clean, production-ready repository for Hubble tension analysis.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run validation (after implementing test)
python tests/test_pipeline.py
```

## Structure

- `src/` - Analysis scripts by phase
- `data/` - Input data files
- `results/` - Output files
- `reference/` - Expected results for verification
- `SAID/` - State Audit Integrity Documentation

## Key Results

- **Phase D Result:** H₀ = 67.57 ± 0.93 km/s/Mpc
- **Tension Reduction:** 97.4%
- **Systematic Fraction:** 51.9%
- **Uncertainty Scaling:** α ≈ +0.9997

## Credits

- Base structure: Claude (Sonnet 4.5)
- Verification: ClaudeCode
- Validation: GPT-5

## Continuous Integration

Automated reproducibility checks run on every push:

- **GitHub Actions**: `.github/workflows/reproducibility.yml`
- **GitLab CI**: `.gitlab-ci.yml`

### Manual Verification

```bash
# Quick check
make verify

# Full test
make test
```

## Repository Structure

```
├── src/              # Analysis scripts by phase
│   ├── phase_a/      # Observer tensor calibration
│   ├── phase_b/      # SN Ia processing
│   ├── phase_c/      # Systematic covariance
│   ├── phase_d/      # Epistemic merge
│   ├── validation/   # Validation scripts
│   └── utils/        # Shared utilities
├── data/             # Input data files
├── results/          # Pipeline outputs
├── reference/        # Expected results
├── SAID/             # State Audit Integrity Documentation
│   ├── audit_logs/   # Session logs and manifests
│   ├── verification/ # Source checksums
│   └── provenance/   # Metadata records
├── scripts/          # Execution scripts
└── tests/            # Test suite
```
